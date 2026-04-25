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
