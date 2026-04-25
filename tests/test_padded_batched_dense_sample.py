"""Tests for :class:`PaddedBatchedDenseSample` and the
:meth:`BatchedDenseSample.to_padded` / :meth:`BatchedDenseSample.from_padded`
roundtrip (ADR-0008 C2)."""

from __future__ import annotations

import pytest
import torch
from torchvision import tv_tensors

from segpaste.types import (
    BatchedDenseSample,
    CameraIntrinsics,
    DenseSample,
    InstanceMask,
    PaddedBatchedDenseSample,
    SemanticMap,
)


def _instance_sample(
    h: int = 16, w: int = 16, num_objects: int = 2, seed: int = 0
) -> DenseSample:
    gen = torch.Generator().manual_seed(seed)
    image = tv_tensors.Image(torch.rand(3, h, w, generator=gen))
    masks = torch.zeros(num_objects, h, w, dtype=torch.bool)
    raw_boxes: list[list[int]] = []
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
        labels=torch.tensor([i + 1 for i in range(num_objects)], dtype=torch.int64),
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


class TestRoundtrip:
    def test_instance_roundtrip_preserves_fields(self) -> None:
        samples = [_instance_sample(num_objects=i + 1, seed=i) for i in range(3)]
        batched = BatchedDenseSample.from_samples(samples)
        padded = batched.to_padded(max_instances=5)

        assert padded.batch_size == 3
        assert padded.max_instances == 5
        assert padded.boxes.shape == (3, 5, 4)
        assert padded.labels.shape == (3, 5)
        assert padded.instance_valid.shape == (3, 5)
        assert padded.instance_masks is not None
        assert padded.instance_masks.shape == (3, 5, 16, 16)
        assert padded.instance_ids is not None
        assert padded.instance_ids.shape == (3, 5)

        expected_valid = torch.tensor(
            [
                [True, False, False, False, False],
                [True, True, False, False, False],
                [True, True, True, False, False],
            ]
        )
        assert torch.equal(padded.instance_valid, expected_valid)

        restored = BatchedDenseSample.from_padded(padded)
        assert restored.batch_size == 3
        for orig_box, back_box in zip(batched.boxes, restored.boxes, strict=True):
            assert torch.equal(
                orig_box.as_subclass(torch.Tensor),
                back_box.as_subclass(torch.Tensor),
            )
            assert back_box.canvas_size == (16, 16)
        for orig_label, back_label in zip(batched.labels, restored.labels, strict=True):
            assert torch.equal(orig_label, back_label)
        assert batched.instance_masks is not None
        assert restored.instance_masks is not None
        for orig_m, back_m in zip(
            batched.instance_masks, restored.instance_masks, strict=True
        ):
            assert torch.equal(
                orig_m.as_subclass(torch.Tensor), back_m.as_subclass(torch.Tensor)
            )
        assert batched.instance_ids is not None
        assert restored.instance_ids is not None
        for orig_id, back_id in zip(
            batched.instance_ids, restored.instance_ids, strict=True
        ):
            assert torch.equal(orig_id, back_id)

    def test_semantic_roundtrip_without_instances(self) -> None:
        samples = [_semantic_sample(seed=i) for i in range(2)]
        batched = BatchedDenseSample.from_samples(samples)
        padded = batched.to_padded(max_instances=3)

        assert padded.instance_masks is None
        assert padded.instance_ids is None
        assert padded.semantic_maps is not None
        assert padded.semantic_maps.shape == (2, 16, 16)
        assert not padded.instance_valid.any()

        restored = BatchedDenseSample.from_padded(padded)
        assert restored.semantic_maps is not None
        assert torch.equal(
            batched.semantic_maps.as_subclass(torch.Tensor),  # pyright: ignore[reportOptionalMemberAccess]
            restored.semantic_maps.as_subclass(torch.Tensor),
        )
        for box in restored.boxes:
            assert box.shape == (0, 4)

    def test_camera_intrinsics_roundtrip(self) -> None:
        sample = _instance_sample(num_objects=1)
        sample = DenseSample(
            image=sample.image,
            boxes=sample.boxes,
            labels=sample.labels,
            instance_ids=sample.instance_ids,
            instance_masks=sample.instance_masks,
            camera_intrinsics=CameraIntrinsics(fx=100.0, fy=200.0, cx=8.0, cy=8.0),
        )
        batched = BatchedDenseSample.from_samples([sample])
        padded = batched.to_padded(max_instances=2)

        assert padded.camera_intrinsics is not None
        assert padded.camera_intrinsics.shape == (1, 4)
        assert torch.allclose(
            padded.camera_intrinsics[0],
            torch.tensor([100.0, 200.0, 8.0, 8.0]),
        )

        restored = BatchedDenseSample.from_padded(padded)
        assert restored.camera_intrinsics is not None
        assert restored.camera_intrinsics[0] == CameraIntrinsics(
            fx=100.0, fy=200.0, cx=8.0, cy=8.0
        )


class TestValiditySemantics:
    def test_padded_rows_are_zero(self) -> None:
        samples = [_instance_sample(num_objects=2, seed=0)]
        padded = BatchedDenseSample.from_samples(samples).to_padded(max_instances=4)

        assert padded.instance_masks is not None
        assert padded.instance_ids is not None
        assert padded.boxes[0, 2:].abs().sum().item() == 0
        assert padded.labels[0, 2:].abs().sum().item() == 0
        assert padded.instance_masks[0, 2:].any().item() is False
        assert padded.instance_ids[0, 2:].abs().sum().item() == 0

    def test_over_budget_raises(self) -> None:
        samples = [_instance_sample(num_objects=4)]
        batched = BatchedDenseSample.from_samples(samples)
        with pytest.raises(ValueError, match="exceeds max_instances"):
            batched.to_padded(max_instances=2)


class TestEdgeCases:
    def test_empty_batch(self) -> None:
        batched = BatchedDenseSample.from_samples([])
        padded = batched.to_padded(max_instances=3)
        assert padded.batch_size == 0
        assert padded.boxes.shape == (0, 3, 4)
        assert padded.labels.shape == (0, 3)
        assert padded.instance_valid.shape == (0, 3)

        restored = BatchedDenseSample.from_padded(padded)
        assert restored.batch_size == 0
        assert restored.boxes == []

    def test_max_instances_zero_accepts_empty_samples(self) -> None:
        samples = [_semantic_sample(seed=i) for i in range(2)]
        batched = BatchedDenseSample.from_samples(samples)
        padded = batched.to_padded(max_instances=0)
        assert padded.boxes.shape == (2, 0, 4)
        assert padded.instance_valid.shape == (2, 0)
        assert padded.max_instances == 0


class TestPostInitValidation:
    def _base(self) -> PaddedBatchedDenseSample:
        samples = [_instance_sample(num_objects=1, seed=0)]
        return BatchedDenseSample.from_samples(samples).to_padded(max_instances=2)

    def test_boxes_wrong_shape_raises(self) -> None:
        base = self._base()
        bad_boxes = torch.zeros((1, 3, 4), dtype=torch.float32)
        with pytest.raises(ValueError, match="boxes must have shape"):
            PaddedBatchedDenseSample(
                images=base.images,
                boxes=bad_boxes,
                labels=base.labels,
                instance_valid=base.instance_valid,
                max_instances=2,
                instance_masks=base.instance_masks,
                instance_ids=base.instance_ids,
            )

    def test_instance_valid_wrong_dtype_raises(self) -> None:
        base = self._base()
        bad_valid = torch.zeros_like(base.instance_valid, dtype=torch.int64)
        with pytest.raises(ValueError, match="instance_valid dtype"):
            PaddedBatchedDenseSample(
                images=base.images,
                boxes=base.boxes,
                labels=base.labels,
                instance_valid=bad_valid,
                max_instances=2,
                instance_masks=base.instance_masks,
                instance_ids=base.instance_ids,
            )

    def test_instance_masks_without_ids_raises(self) -> None:
        base = self._base()
        with pytest.raises(ValueError, match="both be set or both None"):
            PaddedBatchedDenseSample(
                images=base.images,
                boxes=base.boxes,
                labels=base.labels,
                instance_valid=base.instance_valid,
                max_instances=2,
                instance_masks=base.instance_masks,
                instance_ids=None,
            )
