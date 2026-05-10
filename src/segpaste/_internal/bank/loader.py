"""DataLoader integration for the instance-crop bank (ADR-0011 PR4).

Glues an :class:`InstanceBank` to PyTorch's ``Dataset`` x ``Sampler`` x
``DataLoader`` triple so worker-side decode runs in parallel with the
target-image pipeline. Returns batches as pre-staged tensors of shape
``[B, K_bank, C+2, h, w]`` where ``C+2`` packs RGB + alpha + a broadcast
class-id channel — the contract :class:`BankSource` (PR7) consumes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import cast

import torch
from torch.utils.data import DataLoader, Dataset

from segpaste._internal.bank.protocol import BankCrop, InstanceBank
from segpaste._internal.bank.sampler import BankSampler, BankSamplerConfig

Predicate = Callable[[BankCrop], bool]


class BankCocoCropDataset(Dataset[BankCrop]):
    """Dataset wrapper around an :class:`InstanceBank`.

    Optional ``predicate`` runs in the worker before the crop is returned;
    rejected indices walk forward deterministically (``idx + 1 mod len``)
    up to ``max_retries`` times before giving up and emitting the last
    seen crop. ``predicate=None`` is the hot path and adds a single
    ``is None`` check per draw.
    """

    def __init__(
        self,
        bank: InstanceBank,
        predicate: Predicate | None = None,
        *,
        max_retries: int = 8,
    ) -> None:
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")
        self._bank = bank
        self._predicate = predicate
        self._max_retries = max_retries

    def __len__(self) -> int:
        return len(self._bank)

    def __getitem__(self, idx: int) -> BankCrop:
        crop = self._bank[idx]
        if self._predicate is None:
            return crop
        n = len(self._bank)
        for _ in range(self._max_retries):
            if self._predicate(crop):
                return crop
            idx = (idx + 1) % n
            crop = self._bank[idx]
        return crop


def stage_bank_batch(
    crops: Sequence[BankCrop], batch_size: int, k_bank: int
) -> torch.Tensor:
    """Pack a list of ``B*K_bank`` crops into ``[B, K_bank, C+2, h, w]`` float32.

    Channels are RGB (3, in ``[0, 1]``) + alpha (1, ``{0, 1}``) + class-id
    (1, broadcast scalar). The class-id channel uses ``float32`` so the
    consumer's ``where`` / ``gather`` operations stay in one dtype family;
    truncation to int64 happens at the use site if needed.
    """
    expected = batch_size * k_bank
    if len(crops) != expected:
        raise ValueError(
            f"stage_bank_batch expected {expected} crops, got {len(crops)}"
        )
    if expected == 0:
        raise ValueError("batch_size and k_bank must be positive")
    h, w = crops[0].image.shape[-2:]
    out = torch.empty((expected, 5, h, w), dtype=torch.float32)
    for i, crop in enumerate(crops):
        out[i, 0:3] = crop.image.to(torch.float32) / 255.0
        out[i, 3:4] = crop.alpha.to(torch.float32)
        out[i, 4:5] = float(crop.class_id)
    return out.view(batch_size, k_bank, 5, h, w)


def create_bank_dataloader(
    bank: InstanceBank,
    sampler_config: BankSamplerConfig,
    *,
    batch_size: int,
    k_bank: int,
    predicate: Predicate | None = None,
    num_workers: int = 0,
    persistent_workers: bool = False,
    pin_memory: bool = False,
) -> DataLoader[torch.Tensor]:
    """Build a DataLoader that emits pre-staged ``[B, K_bank, C+2, h, w]`` tensors.

    ``sampler_config.samples_per_step`` must equal ``batch_size * k_bank``
    so each DataLoader batch corresponds to one training step.
    """
    if sampler_config.samples_per_step != batch_size * k_bank:
        raise ValueError(
            "sampler_config.samples_per_step "
            f"({sampler_config.samples_per_step}) must equal "
            f"batch_size * k_bank ({batch_size * k_bank})"
        )
    dataset = BankCocoCropDataset(bank, predicate)
    sampler = BankSampler(bank, sampler_config)

    def _collate(crops: list[BankCrop]) -> torch.Tensor:
        return stage_bank_batch(crops, batch_size=batch_size, k_bank=k_bank)

    loader = DataLoader(
        dataset,
        batch_size=batch_size * k_bank,
        sampler=sampler,
        collate_fn=_collate,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    return cast("DataLoader[torch.Tensor]", loader)


__all__ = [
    "BankCocoCropDataset",
    "Predicate",
    "create_bank_dataloader",
    "stage_bank_batch",
]
