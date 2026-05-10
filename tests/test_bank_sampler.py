"""Tests for :class:`BankSampler` (ADR-0011 PR4)."""

from __future__ import annotations

import math

import pytest
import torch

from segpaste._internal.bank import BankCrop
from segpaste._internal.bank.sampler import (
    BankSampler,
    BankSamplerConfig,
    class_distribution_under_sampler,
    repeat_factor_weights,
)


class _SkewedBank:
    """In-memory bank with a controllable per-class crop count."""

    def __init__(self, counts: list[int]) -> None:
        self._h = self._w = 4
        self._classes_per_crop: list[int] = []
        for class_id, count in enumerate(counts):
            self._classes_per_crop.extend([class_id] * count)
        self._n = len(self._classes_per_crop)
        self._classes_t = torch.tensor(self._classes_per_crop, dtype=torch.int64)
        self._freqs = torch.tensor(counts, dtype=torch.int64)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> BankCrop:
        return BankCrop(
            image=torch.zeros((3, self._h, self._w), dtype=torch.uint8),
            alpha=torch.zeros((1, self._h, self._w), dtype=torch.bool),
            class_id=self._classes_per_crop[idx],
            embedding=None,
        )

    @property
    def class_frequencies(self) -> torch.Tensor:
        return self._freqs

    @property
    def crop_class_ids(self) -> torch.Tensor:
        return self._classes_t

    @property
    def crop_size(self) -> tuple[int, int]:
        return (self._h, self._w)

    @property
    def has_embeddings(self) -> bool:
        return False

    @property
    def version(self) -> str:
        return "skewed@deadbeefcafe"


class TestRepeatFactorWeights:
    def test_matches_paper_formula(self) -> None:
        counts = torch.tensor([10000, 1000, 100, 10, 1], dtype=torch.int64)
        t = 1e-3
        r = repeat_factor_weights(counts, t)
        total = float(counts.sum().item())
        for c, n_c in enumerate(counts.tolist()):
            f_c = n_c / total
            expected = max(1.0, math.sqrt(t / f_c))
            assert r[c].item() == pytest.approx(expected, rel=1e-9)

    def test_majority_class_clamped_to_one(self) -> None:
        counts = torch.tensor([1000, 1], dtype=torch.int64)
        r = repeat_factor_weights(counts, t=1e-3)
        assert r[0].item() == pytest.approx(1.0)
        assert r[1].item() > 1.0

    def test_unseen_class_gets_zero_weight(self) -> None:
        counts = torch.tensor([10, 0, 5], dtype=torch.int64)
        r = repeat_factor_weights(counts, t=1e-3)
        assert r[1].item() == 0.0

    def test_empty_bank_raises(self) -> None:
        with pytest.raises(ValueError, match="empty bank"):
            repeat_factor_weights(torch.tensor([0, 0, 0]), t=1e-3)


class TestBankSampler:
    def _config(self, **overrides: int) -> BankSamplerConfig:
        defaults: dict[str, int] = {
            "samples_per_step": 8,
            "num_steps_per_epoch": 4,
        }
        defaults.update(overrides)
        return BankSamplerConfig(**defaults)  # type: ignore[arg-type]

    def test_len_matches_samples_per_step_times_steps(self) -> None:
        bank = _SkewedBank([20, 5, 1])
        sampler = BankSampler(bank, self._config())
        assert len(sampler) == 32

    def test_determinism_same_epoch_same_seed(self) -> None:
        bank = _SkewedBank([20, 5, 1])
        cfg = self._config()
        a = list(BankSampler(bank, cfg))
        b = list(BankSampler(bank, cfg))
        assert a == b

    def test_determinism_changes_with_epoch(self) -> None:
        bank = _SkewedBank([20, 5, 1])
        cfg = self._config(samples_per_step=64, num_steps_per_epoch=2)
        sampler = BankSampler(bank, cfg)
        sampler.set_epoch(0)
        a = list(sampler)
        sampler.set_epoch(1)
        b = list(sampler)
        assert a != b

    def test_ranks_emit_disjoint_streams(self) -> None:
        bank = _SkewedBank([20, 5, 1])
        cfg_a = self._config(num_replicas=4, rank=0, samples_per_step=64)
        cfg_b = self._config(num_replicas=4, rank=1, samples_per_step=64)
        a = list(BankSampler(bank, cfg_a))
        b = list(BankSampler(bank, cfg_b))
        assert a != b

    def test_class_distribution_closer_to_uniform_than_natural(self) -> None:
        # Heavy skew: 90% class 0, 9% class 1, 1% class 2.
        counts = [900, 90, 10]
        bank = _SkewedBank(counts)
        # ``t`` of 1e-3 is calibrated for LVIS-scale datasets; for the
        # ~1k-crop test bank we use a larger value so every minority
        # class lifts above the ``r=1`` clamp and the empirical draw
        # measurably moves toward uniform.
        cfg = BankSamplerConfig(  # type: ignore[call-arg]
            samples_per_step=4096,
            num_steps_per_epoch=1,
            repeat_factor_t=0.5,
        )
        sampler = BankSampler(bank, cfg)
        empirical = class_distribution_under_sampler(
            sampler, bank, num_classes=len(counts)
        )
        empirical_freq = empirical.to(torch.float64) / float(empirical.sum().item())
        natural_freq = bank.class_frequencies.to(torch.float64) / float(
            bank.class_frequencies.sum().item()
        )
        uniform = torch.full_like(empirical_freq, 1.0 / len(counts))
        # The repeat-factor draw should be closer to uniform than the
        # natural distribution under any L1 metric.
        empirical_l1 = float((empirical_freq - uniform).abs().sum().item())
        natural_l1 = float((natural_freq - uniform).abs().sum().item())
        assert empirical_l1 < natural_l1

    def test_negative_epoch_rejected(self) -> None:
        bank = _SkewedBank([10, 1])
        sampler = BankSampler(bank, self._config())
        with pytest.raises(ValueError, match="non-negative"):
            sampler.set_epoch(-1)
