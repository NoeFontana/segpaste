"""Tests for :class:`BatchedDenseSample` (ADR-0004)."""

from __future__ import annotations

import pytest
import torch
from torchvision import tv_tensors

from segpaste.types import (
    BatchedDenseSample,
    DenseSample,
    InstanceMask,
    SemanticMap,
)


def _instance_sample(
    h: int = 16, w: int = 16, num_objects: int = 2, seed: int = 0
) -> DenseSample:
    gen = torch.Generator().manual_seed(seed)
    image = tv_tensors.Image(torch.rand(3, h, w, generator=gen))
    masks = torch.zeros(num_objects, h, w, dtype=torch.bool)
    raw_boxes = []
    for i in range(num_objects):
        x1 = i * 2
        y1 = i * 2
        x2 = x1 + 4
        y2 = y1 + 4
        masks[i, y1:y2, x1:x2] = True
        raw_boxes.append([x1, y1, x2, y2])
    return DenseSample(
        image=image,
        boxes=tv_tensors.BoundingBoxes(  # pyright: ignore[reportCallIssue]
            torch.tensor(raw_boxes, dtype=torch.float32),
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=(h, w),
        ),
        labels=torch.tensor([1] * num_objects, dtype=torch.int64),
        instance_ids=torch.arange(num_objects, dtype=torch.int32),
        instance_masks=InstanceMask(masks),
    )


def _semantic_sample(h: int = 16, w: int = 16, seed: int = 0) -> DenseSample:
    gen = torch.Generator().manual_seed(seed)
    image = tv_tensors.Image(torch.rand(3, h, w, generator=gen))
    sem = torch.randint(0, 4, (h, w), generator=gen, dtype=torch.int64)
    return DenseSample(
        image=image,
        boxes=tv_tensors.BoundingBoxes(  # pyright: ignore[reportCallIssue]
            torch.zeros((0, 4), dtype=torch.float32),
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=(h, w),
        ),
        labels=torch.zeros((0,), dtype=torch.int64),
        semantic_map=SemanticMap(sem),
    )


class TestFromSamples:
    def test_roundtrip_instance(self) -> None:
        samples = [_instance_sample(seed=i) for i in range(3)]
        batched = BatchedDenseSample.from_samples(samples)
        assert batched.batch_size == 3
        assert batched.images.shape == (3, 3, 16, 16)
        assert batched.instance_masks is not None
        assert batched.instance_ids is not None

        restored = batched.to_samples()
        assert len(restored) == 3
        for orig, back in zip(samples, restored, strict=True):
            assert torch.equal(
                orig.image.as_subclass(torch.Tensor),
                back.image.as_subclass(torch.Tensor),
            )
            assert back.instance_masks is not None
            assert orig.instance_masks is not None
            assert torch.equal(
                orig.instance_masks.as_subclass(torch.Tensor),
                back.instance_masks.as_subclass(torch.Tensor),
            )

    def test_roundtrip_semantic(self) -> None:
        samples = [_semantic_sample(seed=i) for i in range(2)]
        batched = BatchedDenseSample.from_samples(samples)
        assert batched.semantic_maps is not None
        assert batched.semantic_maps.shape == (2, 16, 16)
        restored = batched.to_samples()
        assert len(restored) == 2
        for orig, back in zip(samples, restored, strict=True):
            assert back.semantic_map is not None
            assert orig.semantic_map is not None
            assert torch.equal(
                orig.semantic_map.as_subclass(torch.Tensor),
                back.semantic_map.as_subclass(torch.Tensor),
            )


class TestValidation:
    def test_hw_mismatch_raises(self) -> None:
        a = _instance_sample(h=16, w=16)
        b = _instance_sample(h=16, w=24)
        with pytest.raises(ValueError, match="share"):
            BatchedDenseSample.from_samples([a, b])

    def test_modality_mismatch_raises(self) -> None:
        a = _instance_sample()
        b = _semantic_sample()
        with pytest.raises(ValueError, match="modality"):
            BatchedDenseSample.from_samples([a, b])


class TestEmptyBatch:
    def test_empty_from_samples(self) -> None:
        batched = BatchedDenseSample.from_samples([])
        assert batched.batch_size == 0
        assert batched.boxes == []
        assert batched.labels == []
        assert batched.instance_masks is None
        assert batched.instance_ids is None

    def test_empty_to_samples(self) -> None:
        batched = BatchedDenseSample.from_samples([])
        assert batched.to_samples() == []
