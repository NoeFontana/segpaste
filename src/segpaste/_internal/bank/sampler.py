"""Class-balanced bank sampler (ADR-0011 PR4).

Implements Gupta et al. 2019 repeat-factor sampling at the *crop* level::

    f_c = N_c / N_total
    r(c) = max(1, sqrt(t / f_c)),    t ≈ 1e-3
    w_i  ∝ r(c_i)

Subclasses :class:`torch.utils.data.Sampler[int]` (not
:class:`WeightedRandomSampler`) so we get DDP-aware ``rank``/``num_replicas``
slicing and an explicit ``set_epoch`` hook — both needed for stable training
runs across restarts.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch.utils.data import Sampler

from segpaste._internal.bank.protocol import InstanceBank


class BankSamplerConfig(BaseModel):
    """Configuration for :class:`BankSampler`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    base_seed: int = Field(default=0xC0FFEE, ge=0)
    """Base seed mixed with epoch + rank to produce per-iteration RNGs."""

    repeat_factor_t: float = Field(default=1e-3, gt=0.0, le=1.0)
    """``t`` in ``r(c) = max(1, sqrt(t / f_c))``. ``1e-3`` is the
    Gupta 2019 default; lower values ⇒ more aggressive class balancing."""

    samples_per_step: int = Field(gt=0)
    """Indices to draw per training step (typically ``B * K_bank``)."""

    num_steps_per_epoch: int = Field(gt=0)
    """Total training steps per epoch — sampler emits exactly
    ``samples_per_step * num_steps_per_epoch`` indices per epoch."""

    num_replicas: int = Field(default=1, ge=1)
    """Number of DDP ranks. Each rank gets an independent index sequence
    seeded by ``(base_seed, epoch, rank)``."""

    rank: int = Field(default=0, ge=0)
    """This process's DDP rank in ``[0, num_replicas)``."""


def _derive_seed(base_seed: int, epoch: int, rank: int) -> int:
    """Stable 63-bit seed derived from ``(base_seed, epoch, rank)``.

    ``torch.manual_seed`` accepts up to 63-bit unsigned ints. SHA-256 mixing
    keeps the seed sequence well-distributed across small ``base_seed``
    values; ADR-0001 Part (iv) mandates a SHA-256-derived seed for any
    randomness with cross-run reproducibility requirements.
    """
    blob = f"{base_seed}|{epoch}|{rank}".encode()
    digest = hashlib.sha256(blob).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFFFFFFFFFF


def repeat_factor_weights(class_frequencies: torch.Tensor, t: float) -> torch.Tensor:
    """Per-class repeat-factor ``r(c) = max(1, sqrt(t / f_c))``.

    Returns ``float64 [num_classes]`` so downstream multinomial draws keep
    full precision on highly skewed long-tail distributions.
    """
    if class_frequencies.ndim != 1:
        raise ValueError("class_frequencies must be 1-D")
    counts = class_frequencies.to(torch.float64)
    total = counts.sum()
    if total <= 0:
        raise ValueError("class_frequencies sum is zero — empty bank")
    freqs = counts / total
    safe = torch.where(counts > 0, freqs, torch.ones_like(freqs))
    r = torch.clamp(torch.sqrt(t / safe), min=1.0)
    # Classes never seen in the bank get zero weight (no crops to pick).
    r = torch.where(counts > 0, r, torch.zeros_like(r))
    return r


class BankSampler(Sampler[int]):
    """Class-balanced sampler over an :class:`InstanceBank`.

    Computes per-crop weights once at construction (constant for the bank
    lifetime). Per-epoch emits ``samples_per_step * num_steps_per_epoch``
    integer indices via ``torch.multinomial`` seeded by the
    ``(base_seed, epoch, rank)`` triple.
    """

    def __init__(self, bank: InstanceBank, config: BankSamplerConfig) -> None:
        super().__init__()
        if len(bank) == 0:
            raise ValueError("BankSampler requires a non-empty bank")
        self.config = config
        self._epoch = 0
        class_weights = repeat_factor_weights(
            bank.class_frequencies, config.repeat_factor_t
        )
        self._weights = class_weights[bank.crop_class_ids].clone()
        if torch.all(self._weights == 0):
            raise ValueError("repeat-factor weights are all zero — bank misconfigured")

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self._epoch = epoch

    def __len__(self) -> int:
        return self.config.samples_per_step * self.config.num_steps_per_epoch

    def __iter__(self) -> Iterator[int]:
        seed = _derive_seed(self.config.base_seed, self._epoch, self.config.rank)
        gen = torch.Generator()
        gen.manual_seed(seed)
        n = len(self)
        # Multinomial supports up to 2^24 num_samples per call; chunk for
        # safety on multi-million-step epochs.
        chunk = 1 << 20
        remaining = n
        while remaining > 0:
            take = min(chunk, remaining)
            indices = torch.multinomial(
                self._weights, num_samples=take, replacement=True, generator=gen
            )
            yield from indices.tolist()
            remaining -= take


def class_distribution_under_sampler(
    sampler: BankSampler, bank: InstanceBank, num_classes: int
) -> torch.Tensor:
    """Empirical ``[num_classes]`` int64 histogram of one epoch's draws.

    Helper for tests and observability — not on the hot path.
    """
    counts = torch.zeros((num_classes,), dtype=torch.int64)
    for idx in sampler:
        counts[bank[idx].class_id] += 1
    return counts


__all__ = [
    "BankSampler",
    "BankSamplerConfig",
    "class_distribution_under_sampler",
    "repeat_factor_weights",
]
