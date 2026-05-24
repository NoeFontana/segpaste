"""torchvision ``collate_fn`` for :class:`BatchCopyPaste` (ADR-0015)."""

from __future__ import annotations

from collections.abc import Callable

from segpaste.types import (
    BatchedDenseSample,
    DenseSample,
    PaddedBatchedDenseSample,
)

CollateFn = Callable[[list[DenseSample]], PaddedBatchedDenseSample]


def make_segpaste_collate_fn(max_instances: int = 32) -> CollateFn:
    """Return a ``collate_fn`` mapping ``list[DenseSample]`` → padded batch.

    Use as ``DataLoader(dataset, collate_fn=make_segpaste_collate_fn(K))``.
    The returned :class:`PaddedBatchedDenseSample` is the input shape of
    :meth:`segpaste.BatchCopyPaste.forward`.
    """

    def _collate(samples: list[DenseSample]) -> PaddedBatchedDenseSample:
        return BatchedDenseSample.from_samples(samples).to_padded(max_instances)

    return _collate
