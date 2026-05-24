"""ADR-0015 §1: lazy-import gate raises a helpful ImportError."""

from __future__ import annotations

import sys

import pytest
from torch.utils.data import Dataset

from segpaste import make_segpaste_datamodule
from segpaste.types import DenseSample


class _EmptyDataset(Dataset[DenseSample]):
    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx: int) -> DenseSample:
        raise IndexError(idx)


def test_missing_lightning_raises_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``lightning`` is unimportable, the factory points at the extra."""
    monkeypatch.setitem(sys.modules, "lightning", None)
    monkeypatch.setitem(sys.modules, "lightning.pytorch", None)

    with pytest.raises(ImportError, match=r"pip install 'segpaste\[lightning\]'"):
        make_segpaste_datamodule("coco-panoptic", train_dataset=_EmptyDataset())
