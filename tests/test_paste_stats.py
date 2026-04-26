"""Unit tests for ``compute_paste_stats``."""

from __future__ import annotations

from dataclasses import replace

import torch
from torchvision import tv_tensors

from segpaste._internal.viz.paste_stats import compute_paste_stats
from segpaste._internal.viz.synthetic import make_synthetic_samples
from segpaste.types import DenseSample, InstanceMask


def _make_after_with_pasted_id(before: DenseSample, new_id: int) -> DenseSample:
    """Append one new instance (a known-area square) to *before*."""
    h, w = before.image.shape[-2:]
    new_mask = torch.zeros((1, h, w), dtype=torch.bool)
    new_mask[0, 0:8, 0:8] = True

    assert before.instance_masks is not None
    assert before.instance_ids is not None
    masks = torch.cat(
        [before.instance_masks.as_subclass(torch.Tensor), new_mask], dim=0
    )
    ids = torch.cat([before.instance_ids, torch.tensor([new_id], dtype=torch.int32)])
    labels = torch.cat([before.labels, torch.tensor([99], dtype=torch.int64)])
    boxes = tv_tensors.BoundingBoxes(  # pyright: ignore[reportCallIssue]
        torch.cat(
            [
                before.boxes.as_subclass(torch.Tensor),
                torch.tensor([[0.0, 0.0, 8.0, 8.0]]),
            ]
        ),
        format=tv_tensors.BoundingBoxFormat.XYXY,
        canvas_size=(h, w),
    )
    return replace(
        before,
        boxes=boxes,
        labels=labels,
        instance_ids=ids,
        instance_masks=InstanceMask(masks),
    )


def test_returns_none_when_instance_modality_absent() -> None:
    sample = make_synthetic_samples(seed=0, count=1)[0]
    h, w = int(sample.image.shape[-2]), int(sample.image.shape[-1])
    sans_instance = replace(
        sample,
        instance_ids=None,
        instance_masks=None,
        labels=sample.labels[:0],
        boxes=tv_tensors.BoundingBoxes(  # pyright: ignore[reportCallIssue]
            torch.zeros((0, 4), dtype=torch.float32),
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=(h, w),
        ),
    )

    assert compute_paste_stats(sans_instance, sample) is None
    assert compute_paste_stats(sample, sans_instance) is None


def test_zero_pastes_returns_zero_area() -> None:
    sample = make_synthetic_samples(seed=0, count=1)[0]

    stats = compute_paste_stats(sample, sample)

    assert stats is not None
    assert stats.K_pasted == 0
    assert stats.paste_area_frac == 0.0


def test_one_paste_reports_correct_area_fraction() -> None:
    before = make_synthetic_samples(seed=0, count=1)[0]
    new_id = (
        int(before.instance_ids.max().item()) + 1
        if before.instance_ids is not None
        else 99
    )
    after = _make_after_with_pasted_id(before, new_id=new_id)

    stats = compute_paste_stats(before, after)

    assert stats is not None
    assert stats.K_pasted == 1
    h, w = after.image.shape[-2:]
    expected = 64.0 / float(h * w)  # the 8x8 square added by the helper
    assert abs(stats.paste_area_frac - expected) < 1e-9
