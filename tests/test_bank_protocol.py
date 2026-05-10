"""Structural conformance tests for :class:`InstanceBank` (ADR-0011 PR4)."""

from __future__ import annotations

import pytest
import torch

from segpaste import InstanceBank
from segpaste._internal.bank import BankCrop


class _ToyBank:
    """Minimal in-memory bank used to exercise the Protocol contract."""

    def __init__(self, n: int = 8, h: int = 16, w: int = 16, num_classes: int = 4):
        self._h = h
        self._w = w
        self._classes = torch.randint(0, num_classes, (n,), dtype=torch.int64)
        self._crops = [
            BankCrop(
                image=torch.randint(0, 256, (3, h, w), dtype=torch.uint8),
                alpha=torch.zeros((1, h, w), dtype=torch.bool),
                class_id=int(self._classes[i].item()),
                embedding=None,
            )
            for i in range(n)
        ]
        self._freqs = torch.bincount(self._classes, minlength=num_classes)

    def __len__(self) -> int:
        return len(self._crops)

    def __getitem__(self, idx: int) -> BankCrop:
        return self._crops[idx]

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


def test_toy_bank_satisfies_protocol() -> None:
    bank: InstanceBank = _ToyBank()
    assert isinstance(bank, InstanceBank)


def test_protocol_is_runtime_checkable_and_imported_publicly() -> None:
    import segpaste

    assert "InstanceBank" in segpaste.__all__
    assert hasattr(segpaste, "InstanceBank")


def test_getitem_returns_bankcrop_named_tuple() -> None:
    bank = _ToyBank()
    crop = bank[0]
    assert isinstance(crop, BankCrop)
    assert crop.image.dtype == torch.uint8
    assert crop.alpha.dtype == torch.bool
    assert isinstance(crop.class_id, int)


def test_class_frequencies_sum_to_len() -> None:
    bank = _ToyBank(n=12, num_classes=5)
    assert int(bank.class_frequencies.sum().item()) == len(bank)


def test_crop_class_ids_matches_getitem() -> None:
    bank = _ToyBank(n=10)
    ids = bank.crop_class_ids
    for i in range(len(bank)):
        assert int(ids[i].item()) == bank[i].class_id


def test_index_out_of_range_raises() -> None:
    bank = _ToyBank(n=4)
    with pytest.raises(IndexError):
        bank[99]
