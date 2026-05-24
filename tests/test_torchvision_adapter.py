"""ADR-0015 §1: torchvision adapter — collate_fn shape & types."""

from __future__ import annotations

from segpaste import (
    BatchCopyPaste,
    PaddedBatchedDenseSample,
    get_preset,
    make_segpaste_collate_fn,
)
from tests.shared import make_disjoint_panoptic_sample


def test_collate_fn_returns_padded_batched_dense_sample() -> None:
    collate = make_segpaste_collate_fn(max_instances=8)
    samples = [make_disjoint_panoptic_sample(i) for i in range(3)]
    padded = collate(samples)

    assert isinstance(padded, PaddedBatchedDenseSample)
    assert padded.images.shape == (3, 3, 24, 24)
    assert padded.boxes.shape == (3, 8, 4)
    assert padded.labels.shape == (3, 8)
    assert padded.instance_valid.shape == (3, 8)
    assert padded.instance_valid[:, :2].all().item()
    assert not padded.instance_valid[:, 2:].any().item()


def test_collate_fn_feeds_batch_copy_paste() -> None:
    """The torchvision collate_fn closes the loop into :class:`BatchCopyPaste`."""
    collate = make_segpaste_collate_fn(max_instances=8)
    samples = [make_disjoint_panoptic_sample(i) for i in range(4)]
    padded = collate(samples)

    augment = BatchCopyPaste(get_preset("coco-instance").batch_copy_paste)
    out = augment(padded)

    assert isinstance(out, PaddedBatchedDenseSample)
    assert out.images.shape == padded.images.shape
