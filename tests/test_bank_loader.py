"""Smoke tests for :func:`create_bank_dataloader` + ``stage_bank_batch``."""

from __future__ import annotations

import pytest
import torch

from segpaste._internal.bank import BankCrop
from segpaste._internal.bank.loader import (
    BankCocoCropDataset,
    create_bank_dataloader,
    stage_bank_batch,
)
from segpaste._internal.bank.sampler import BankSamplerConfig


class _ToyBank:
    def __init__(self, n: int = 16, h: int = 8, w: int = 8) -> None:
        self._n = n
        self._h = h
        self._w = w
        self._classes = torch.arange(n, dtype=torch.int64) % 3
        self._freqs = torch.bincount(self._classes, minlength=3)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> BankCrop:
        return BankCrop(
            image=torch.full((3, self._h, self._w), idx, dtype=torch.uint8),
            alpha=torch.full((1, self._h, self._w), bool(idx % 2), dtype=torch.bool),
            class_id=int(self._classes[idx].item()),
            embedding=None,
        )

    @property
    def class_frequencies(self) -> torch.Tensor:
        return self._freqs

    @property
    def crop_class_ids(self) -> torch.Tensor:
        return self._classes

    @property
    def crop_size(self) -> tuple[int, int]:
        return (self._h, self._w)

    @property
    def has_embeddings(self) -> bool:
        return False

    @property
    def version(self) -> str:
        return "toy@deadbeefcafe"


def test_stage_bank_batch_shapes_and_dtype() -> None:
    bank = _ToyBank(h=4, w=4)
    crops = [bank[i] for i in range(6)]
    out = stage_bank_batch(crops, batch_size=2, k_bank=3)
    assert out.shape == (2, 3, 5, 4, 4)
    assert out.dtype == torch.float32
    # Image channel scaled to [0, 1].
    assert out[0, 0, 0:3].max() <= 1.0
    # Alpha channel is bool-ish.
    assert torch.all((out[..., 3:4, :, :] == 0.0) | (out[..., 3:4, :, :] == 1.0))
    # Class-id channel matches the crop's class id (broadcast scalar).
    assert out[0, 0, 4, 0, 0].item() == float(crops[0].class_id)


def test_stage_bank_batch_size_mismatch_raises() -> None:
    bank = _ToyBank()
    crops = [bank[i] for i in range(5)]
    with pytest.raises(ValueError, match="expected"):
        stage_bank_batch(crops, batch_size=2, k_bank=3)


def test_predicate_filters_in_dataset() -> None:
    bank = _ToyBank(n=8)

    def even_class(crop: BankCrop) -> bool:
        return crop.class_id % 2 == 0

    dataset = BankCocoCropDataset(bank, even_class)
    crop = dataset[1]
    assert crop.class_id % 2 == 0


def test_predicate_falls_back_after_max_retries() -> None:
    bank = _ToyBank(n=4)
    dataset = BankCocoCropDataset(bank, lambda _c: False, max_retries=3)
    crop = dataset[0]
    assert isinstance(crop, BankCrop)


def test_create_bank_dataloader_yields_staged_tensor() -> None:
    bank = _ToyBank(n=12)
    cfg = BankSamplerConfig(  # type: ignore[call-arg]
        samples_per_step=6,
        num_steps_per_epoch=2,
    )
    loader = create_bank_dataloader(
        bank,
        cfg,
        batch_size=2,
        k_bank=3,
        num_workers=0,
    )
    batches = list(loader)
    assert len(batches) == 2
    for b in batches:
        assert b.shape == (2, 3, 5, 8, 8)
        assert b.dtype == torch.float32


def test_create_bank_dataloader_size_mismatch_raises() -> None:
    bank = _ToyBank(n=12)
    cfg = BankSamplerConfig(  # type: ignore[call-arg]
        samples_per_step=8,  # B*K_bank should be 6
        num_steps_per_epoch=1,
    )
    with pytest.raises(ValueError, match="samples_per_step"):
        create_bank_dataloader(bank, cfg, batch_size=2, k_bank=3)
