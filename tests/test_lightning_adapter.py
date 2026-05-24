"""ADR-0015 §1: Lightning DataModule factory smoke test."""

from __future__ import annotations

import pytest
from torch.utils.data import Dataset

from segpaste import PaddedBatchedDenseSample, make_segpaste_datamodule
from segpaste.types import DenseSample
from tests.shared import make_disjoint_panoptic_sample

pytest.importorskip("lightning")


class _InMemoryDataset(Dataset[DenseSample]):
    """Minimal ``torch.utils.data.Dataset``-compatible wrapper."""

    def __init__(self, samples: list[DenseSample]) -> None:
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> DenseSample:
        return self._samples[idx]


def test_datamodule_train_dataloader_emits_padded_batch() -> None:
    samples = [make_disjoint_panoptic_sample(i) for i in range(4)]
    dm = make_segpaste_datamodule(
        "coco-panoptic",
        _InMemoryDataset(samples),
        batch_size=2,
        max_instances=8,
        num_workers=0,
    )

    dm.setup()
    loader = dm.train_dataloader()
    batch = next(iter(loader))

    assert isinstance(batch, PaddedBatchedDenseSample)
    assert batch.images.shape == (2, 3, 24, 24)
    assert batch.instance_valid.shape == (2, 8)
