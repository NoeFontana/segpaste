"""Tests for :class:`BatchedPlacementSampler` (ADR-0008 C3)."""

from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError
from torchvision import tv_tensors

from segpaste._internal.gpu.batched_placement import (
    BatchedPlacement,
    BatchedPlacementConfig,
    BatchedPlacementSampler,
)
from segpaste.types import (
    BatchedDenseSample,
    DenseSample,
    InstanceMask,
    PaddedBatchedDenseSample,
)


def _instance_sample(
    h: int = 32, w: int = 32, num_objects: int = 3, seed: int = 0
) -> DenseSample:
    gen = torch.Generator().manual_seed(seed)
    image = tv_tensors.Image(torch.rand(3, h, w, generator=gen))
    masks = torch.zeros(num_objects, h, w, dtype=torch.bool)
    raw_boxes: list[list[int]] = []
    for i in range(num_objects):
        x1 = i * 4
        y1 = i * 4
        x2 = x1 + 6
        y2 = y1 + 6
        masks[i, y1:y2, x1:x2] = True
        raw_boxes.append([x1, y1, x2, y2])
    return DenseSample(
        image=image,
        boxes=tv_tensors.BoundingBoxes(  # pyright: ignore[reportCallIssue]
            torch.tensor(raw_boxes, dtype=torch.float32),
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=(h, w),
        ),
        labels=torch.arange(1, num_objects + 1, dtype=torch.int64),
        instance_ids=torch.arange(num_objects, dtype=torch.int32),
        instance_masks=InstanceMask(masks),
    )


def _padded(
    b: int = 4, h: int = 32, w: int = 32, k: int = 5
) -> PaddedBatchedDenseSample:
    samples = [
        _instance_sample(h=h, w=w, num_objects=2 + (i % 2), seed=i) for i in range(b)
    ]
    return BatchedDenseSample.from_samples(samples).to_padded(max_instances=k)


def _sampler(scale_range: tuple[float, float] = (1.0, 1.0)) -> BatchedPlacementSampler:
    return BatchedPlacementSampler(
        BatchedPlacementConfig(scale_range=scale_range, hflip_probability=0.5)
    )


class TestShapesAndTypes:
    def test_output_shapes(self) -> None:
        padded = _padded(b=4, k=5)
        sampler = _sampler()
        out = sampler(padded, torch.Generator().manual_seed(0))
        assert isinstance(out, BatchedPlacement)
        assert out.source_idx.shape == (4,)
        assert out.translate.shape == (4, 2)
        assert out.scale.shape == (4,)
        assert out.hflip.shape == (4,)
        assert out.paste_valid.shape == (4, 5)
        assert out.source_idx.dtype == torch.int64
        assert out.hflip.dtype == torch.bool
        assert out.paste_valid.dtype == torch.bool


class TestDiagonalMask:
    def test_source_idx_never_equals_target_idx(self) -> None:
        padded = _padded(b=8)
        sampler = _sampler()
        for seed in range(20):
            out = sampler(padded, torch.Generator().manual_seed(seed))
            target_idx = torch.arange(8, dtype=torch.int64)
            assert bool((out.source_idx != target_idx).all())


class TestValidityGating:
    def test_invalid_source_rows_gate_paste_valid(self) -> None:
        padded = _padded(b=4, k=5)
        sampler = _sampler()
        out = sampler(padded, torch.Generator().manual_seed(0))
        for i in range(4):
            j = int(out.source_idx[i].item())
            assert bool((out.paste_valid[i] <= padded.instance_valid[j]).all())

    def test_oversized_scale_marks_invalid(self) -> None:
        padded = _padded(b=4, h=32, w=32, k=5)
        sampler = _sampler(scale_range=(20.0, 20.0))
        out = sampler(padded, torch.Generator().manual_seed(0))
        assert not bool(out.paste_valid.any())


class TestBoundsInvariant:
    def test_translate_within_bounds_per_sample(self) -> None:
        padded = _padded(b=4, h=32, w=32, k=5)
        sampler = _sampler(scale_range=(0.5, 1.5))
        out = sampler(padded, torch.Generator().manual_seed(0))
        source_boxes = padded.boxes[out.source_idx]
        box_wh = source_boxes[..., 2:] - source_boxes[..., :2]
        scaled_wh_per_slot = box_wh * out.scale.unsqueeze(-1).unsqueeze(-1)
        valid_mask = out.paste_valid
        max_slot_h = scaled_wh_per_slot[..., 1].amax(dim=-1)
        max_slot_w = scaled_wh_per_slot[..., 0].amax(dim=-1)
        has_any_valid = valid_mask.any(dim=-1)
        assert bool((out.translate[..., 0][has_any_valid] >= 0.0).all())
        assert bool((out.translate[..., 1][has_any_valid] >= 0.0).all())
        assert bool(
            (
                out.translate[..., 0][has_any_valid] + max_slot_h[has_any_valid]
                <= 32.0 + 1e-4
            ).all()
        )
        assert bool(
            (
                out.translate[..., 1][has_any_valid] + max_slot_w[has_any_valid]
                <= 32.0 + 1e-4
            ).all()
        )


class TestReproducibility:
    def test_matched_seed_matches(self) -> None:
        padded = _padded(b=4, k=5)
        sampler = _sampler(scale_range=(0.5, 1.5))
        a = sampler(padded, torch.Generator().manual_seed(42))
        b = sampler(padded, torch.Generator().manual_seed(42))
        assert torch.equal(a.source_idx, b.source_idx)
        assert torch.equal(a.translate, b.translate)
        assert torch.equal(a.scale, b.scale)
        assert torch.equal(a.hflip, b.hflip)
        assert torch.equal(a.paste_valid, b.paste_valid)


class TestEdgeCases:
    def test_b_zero_returns_empty_placement(self) -> None:
        padded = BatchedDenseSample.from_samples([]).to_padded(max_instances=3)
        sampler = _sampler()
        out = sampler(padded, torch.Generator().manual_seed(0))
        assert out.source_idx.numel() == 0
        assert out.paste_valid.shape == (0, 3)

    def test_b_one_emits_all_invalid(self) -> None:
        samples = [_instance_sample()]
        padded = BatchedDenseSample.from_samples(samples).to_padded(max_instances=3)
        sampler = _sampler()
        out = sampler(padded, torch.Generator().manual_seed(0))
        assert not bool(out.paste_valid.any())


class TestPydanticConfig:
    def test_extra_forbid(self) -> None:
        with pytest.raises(ValidationError):
            BatchedPlacementConfig(bogus_field=1.0)  # pyright: ignore[reportCallIssue]


def _small_box_sample(seed: int) -> DenseSample:
    """Single 6x6 mask anchored at (2, 2) in a 32x32 canvas — fits in [0, 16]^2."""
    h = w = 32
    gen = torch.Generator().manual_seed(seed)
    image = tv_tensors.Image(torch.rand(3, h, w, generator=gen))
    mask = torch.zeros(1, h, w, dtype=torch.bool)
    mask[0, 2:8, 2:8] = True
    return DenseSample(
        image=image,
        boxes=tv_tensors.BoundingBoxes(  # pyright: ignore[reportCallIssue]
            torch.tensor([[2.0, 2.0, 8.0, 8.0]]),
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=(h, w),
        ),
        labels=torch.tensor([1], dtype=torch.int64),
        instance_ids=torch.tensor([0], dtype=torch.int32),
        instance_masks=InstanceMask(mask),
    )


def _wide_box_sample(seed: int) -> DenseSample:
    """Single mask whose box max=20 exceeds the 16-px valid extent."""
    h = w = 32
    gen = torch.Generator().manual_seed(seed)
    image = tv_tensors.Image(torch.rand(3, h, w, generator=gen))
    mask = torch.zeros(1, h, w, dtype=torch.bool)
    mask[0, 0:20, 0:20] = True
    return DenseSample(
        image=image,
        boxes=tv_tensors.BoundingBoxes(  # pyright: ignore[reportCallIssue]
            torch.tensor([[0.0, 0.0, 20.0, 20.0]]),
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=(h, w),
        ),
        labels=torch.tensor([1], dtype=torch.int64),
        instance_ids=torch.tensor([0], dtype=torch.int32),
        instance_masks=InstanceMask(mask),
    )


class TestValidExtent:
    def test_translate_respects_valid_extent(self) -> None:
        padded = _padded(b=4, h=32, w=32, k=5)
        sampler = _sampler(scale_range=(0.5, 1.0))
        ve = torch.full((4, 2), 16.0)
        out = sampler(padded, torch.Generator().manual_seed(0), valid_extent=ve)

        source_boxes = padded.boxes[out.source_idx]
        box_wh = source_boxes[..., 2:] - source_boxes[..., :2]
        scaled_wh = box_wh * out.scale.view(-1, 1, 1)
        max_slot_h = scaled_wh[..., 1].amax(dim=-1)
        max_slot_w = scaled_wh[..., 0].amax(dim=-1)
        has_any = out.paste_valid.any(dim=-1)
        eps = 1e-4

        if has_any.any():
            assert bool((out.translate[..., 0][has_any] >= 0.0).all())
            assert bool((out.translate[..., 1][has_any] >= 0.0).all())
            assert bool(
                (
                    out.translate[..., 0][has_any] + max_slot_h[has_any] <= 16.0 + eps
                ).all()
            )
            assert bool(
                (
                    out.translate[..., 1][has_any] + max_slot_w[has_any] <= 16.0 + eps
                ).all()
            )

    def test_source_bbox_outside_valid_extent_clears_paste_valid(self) -> None:
        """A source instance whose bbox extends past valid_extent must not paste."""
        # B=2 → diagonal mask forces source_idx[1] = 0, source_idx[0] = 1.
        # hflip=0 isolates the source-side gate from the target-side fits gate
        # (which rejects flipped placements when (w-1)-bx1 exceeds valid_extent).
        samples = [_wide_box_sample(seed=0), _small_box_sample(seed=1)]
        padded = BatchedDenseSample.from_samples(samples).to_padded(max_instances=1)
        sampler = BatchedPlacementSampler(
            BatchedPlacementConfig(scale_range=(1.0, 1.0), hflip_probability=0.0)
        )
        ve = torch.tensor([[16.0, 16.0], [16.0, 16.0]])
        out = sampler(padded, torch.Generator().manual_seed(0), valid_extent=ve)

        # Target 1 sources from sample 0 (bbox max 20 > 16) — paste_valid cleared.
        assert int(out.source_idx[1].item()) == 0
        assert not bool(out.paste_valid[1, 0].item())
        # Target 0 sources from sample 1 (bbox max 8 ≤ 16) — paste_valid preserved.
        assert int(out.source_idx[0].item()) == 1
        assert bool(out.paste_valid[0, 0].item())

    def test_none_recovers_full_canvas_behavior(self) -> None:
        """``valid_extent=None`` must be bitwise-identical to the prior signature."""
        padded = _padded(b=4, h=32, w=32, k=5)
        sampler = _sampler(scale_range=(0.5, 1.5))
        a = sampler(padded, torch.Generator().manual_seed(7), valid_extent=None)
        b = sampler(padded, torch.Generator().manual_seed(7))
        assert torch.equal(a.source_idx, b.source_idx)
        assert torch.equal(a.translate, b.translate)
        assert torch.equal(a.scale, b.scale)
        assert torch.equal(a.hflip, b.hflip)
        assert torch.equal(a.paste_valid, b.paste_valid)


class TestPasteProb:
    def test_paste_prob_zero_clears_all(self) -> None:
        padded = _padded(b=8, k=5)
        sampler = BatchedPlacementSampler(
            BatchedPlacementConfig(scale_range=(1.0, 1.0), paste_prob=0.0)
        )
        out = sampler(padded, torch.Generator().manual_seed(0))
        assert not bool(out.paste_valid.any())

    def test_paste_prob_one_is_no_op(self) -> None:
        padded = _padded(b=4, k=5)
        baseline = BatchedPlacementSampler(
            BatchedPlacementConfig(scale_range=(1.0, 1.0))
        )
        gated = BatchedPlacementSampler(
            BatchedPlacementConfig(scale_range=(1.0, 1.0), paste_prob=1.0)
        )
        a = baseline(padded, torch.Generator().manual_seed(7))
        b = gated(padded, torch.Generator().manual_seed(7))
        assert torch.equal(a.paste_valid, b.paste_valid)


class TestKRange:
    def test_k_range_truncates_to_cap(self) -> None:
        padded = _padded(b=4, k=5)
        sampler = BatchedPlacementSampler(
            BatchedPlacementConfig(scale_range=(1.0, 1.0), k_range=(2, 2))
        )
        out = sampler(padded, torch.Generator().manual_seed(0))
        per_image = out.paste_valid.sum(dim=-1)
        assert bool((per_image <= 2).all())

    def test_k_range_zero_clears_all(self) -> None:
        padded = _padded(b=4, k=5)
        sampler = BatchedPlacementSampler(
            BatchedPlacementConfig(scale_range=(1.0, 1.0), k_range=(0, 0))
        )
        out = sampler(padded, torch.Generator().manual_seed(0))
        assert not bool(out.paste_valid.any())


class TestSourceEligible:
    def test_ineligible_sources_clear_paste_valid(self) -> None:
        padded = _padded(b=4, k=5)
        # Mark all source rows as ineligible — paste_valid must collapse.
        ineligible = torch.zeros_like(padded.instance_valid)
        sampler = _sampler()
        out = sampler(
            padded,
            torch.Generator().manual_seed(0),
            source_eligible=ineligible,
        )
        assert not bool(out.paste_valid.any())

    def test_none_recovers_default_behavior(self) -> None:
        padded = _padded(b=4, k=5)
        sampler = _sampler(scale_range=(0.5, 1.5))
        a = sampler(padded, torch.Generator().manual_seed(11), source_eligible=None)
        b = sampler(padded, torch.Generator().manual_seed(11))
        assert torch.equal(a.paste_valid, b.paste_valid)
