"""Tests for :class:`AffinePropagator` (ADR-0008 C4)."""

from __future__ import annotations

import torch
from torchvision import tv_tensors

from segpaste._internal.gpu.affine_propagate import AffinePropagator
from segpaste._internal.gpu.batched_placement import BatchedPlacement
from segpaste.types import (
    BatchedDenseSample,
    CameraIntrinsics,
    DenseSample,
    InstanceMask,
    PaddedBatchedDenseSample,
    PanopticMap,
    SemanticMap,
)

H = W = 32


def _sample(
    num_objects: int = 2,
    seed: int = 0,
    with_semantic: bool = False,
    with_panoptic: bool = False,
    with_depth: bool = False,
    with_normals: bool = False,
) -> DenseSample:
    gen = torch.Generator().manual_seed(seed)
    image = tv_tensors.Image(torch.rand(3, H, W, generator=gen))
    masks = torch.zeros(num_objects, H, W, dtype=torch.bool)
    raw_boxes: list[list[int]] = []
    for i in range(num_objects):
        x1 = 2 + i * 4
        y1 = 2 + i * 4
        x2 = x1 + 5
        y2 = y1 + 5
        masks[i, y1:y2, x1:x2] = True
        raw_boxes.append([x1, y1, x2, y2])
    fields: dict[str, object] = {
        "image": image,
        "boxes": tv_tensors.BoundingBoxes(  # pyright: ignore[reportCallIssue]
            torch.tensor(raw_boxes, dtype=torch.float32),
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=(H, W),
        ),
        "labels": torch.arange(1, num_objects + 1, dtype=torch.int64),
        "instance_ids": torch.arange(num_objects, dtype=torch.int32),
        "instance_masks": InstanceMask(masks),
    }
    if with_semantic:
        fields["semantic_map"] = SemanticMap(
            torch.randint(0, 4, (H, W), generator=gen, dtype=torch.int64)
        )
    if with_panoptic:
        fields["panoptic_map"] = PanopticMap(
            torch.randint(0, 8, (H, W), generator=gen, dtype=torch.int64)
        )
    if with_depth:
        fields["depth"] = torch.rand(1, H, W, generator=gen) * 10.0
        fields["depth_valid"] = torch.rand(1, H, W, generator=gen) > 0.1
        fields["camera_intrinsics"] = CameraIntrinsics(
            fx=100.0, fy=100.0, cx=W / 2.0, cy=H / 2.0
        )
    if with_normals:
        raw = torch.randn(3, H, W, generator=gen)
        fields["normals"] = raw / raw.norm(dim=0, keepdim=True).clamp(min=1e-6)
    return DenseSample(**fields)  # pyright: ignore[reportArgumentType]


def _padded(b: int = 2, k: int = 3, **kwargs: bool) -> PaddedBatchedDenseSample:
    samples = [_sample(num_objects=2, seed=i, **kwargs) for i in range(b)]
    return BatchedDenseSample.from_samples(samples).to_padded(max_instances=k)


def _identity_placement(b: int, k: int, device: torch.device) -> BatchedPlacement:
    return BatchedPlacement(
        source_idx=torch.tensor([(i + 1) % b for i in range(b)], device=device),
        translate=torch.zeros((b, 2), device=device),
        scale=torch.ones((b,), device=device),
        hflip=torch.zeros((b,), dtype=torch.bool, device=device),
        paste_valid=torch.ones((b, k), dtype=torch.bool, device=device),
    )


class TestShapeAndOutputType:
    def test_output_is_padded_with_all_modalities(self) -> None:
        padded = _padded(
            b=2, k=3, with_semantic=True, with_depth=True, with_normals=True
        )
        placement = _identity_placement(2, 3, padded.images.device)
        out = AffinePropagator()(padded, placement)
        assert isinstance(out, PaddedBatchedDenseSample)
        assert out.images.shape == (2, 3, H, W)
        assert out.boxes.shape == (2, 3, 4)
        assert out.instance_masks is not None
        assert out.instance_masks.shape == (2, 3, H, W)
        assert out.instance_masks.dtype == torch.bool
        assert out.semantic_maps is not None and out.semantic_maps.shape == (2, H, W)
        assert out.depth is not None and out.depth.shape == (2, 1, H, W)
        assert out.depth_valid is not None and out.depth_valid.dtype == torch.bool
        assert out.normals is not None and out.normals.shape == (2, 3, H, W)


class TestIdentityAffine:
    def test_identity_image_equals_source(self) -> None:
        padded = _padded(b=2, k=3)
        placement = _identity_placement(2, 3, padded.images.device)
        out = AffinePropagator()(padded, placement)
        src_images = padded.images.as_subclass(torch.Tensor)[placement.source_idx]
        out_t = out.images.as_subclass(torch.Tensor)
        # Bilinear grid_sample with align_corners=False on an integer-grid
        # identity should reproduce the source within tiny numeric error.
        assert torch.allclose(out_t, src_images, atol=1e-5)

    def test_identity_masks_preserve_cardinality(self) -> None:
        padded = _padded(b=2, k=3)
        placement = _identity_placement(2, 3, padded.images.device)
        out = AffinePropagator()(padded, placement)
        assert out.instance_masks is not None
        assert padded.instance_masks is not None
        src_masks = padded.instance_masks[placement.source_idx]
        assert out.instance_masks.dtype == torch.bool
        assert torch.equal(out.instance_masks, src_masks)


class TestTranslation:
    def test_pure_translation_shifts_mask(self) -> None:
        padded = _padded(b=2, k=3)
        b = 2
        placement = BatchedPlacement(
            source_idx=torch.tensor([1, 0]),
            translate=torch.tensor([[5.0, 4.0], [0.0, 0.0]]),
            scale=torch.ones(b),
            hflip=torch.zeros(b, dtype=torch.bool),
            paste_valid=torch.ones((b, 3), dtype=torch.bool),
        )
        out = AffinePropagator()(padded, placement)
        assert out.instance_masks is not None
        src = padded.instance_masks  # pyright: ignore[reportOptionalMemberAccess]
        assert src is not None
        src_0 = src[1, 0]
        out_0 = out.instance_masks[0, 0]
        src_ys, src_xs = src_0.nonzero(as_tuple=True)
        out_ys, out_xs = out_0.nonzero(as_tuple=True)
        assert out_ys.numel() > 0
        assert int(out_ys.min().item()) == int(src_ys.min().item()) + 5
        assert int(out_xs.min().item()) == int(src_xs.min().item()) + 4

    def test_translated_boxes_shift_correctly(self) -> None:
        padded = _padded(b=2, k=3)
        b = 2
        placement = BatchedPlacement(
            source_idx=torch.tensor([1, 0]),
            translate=torch.tensor([[5.0, 4.0], [0.0, 0.0]]),
            scale=torch.ones(b),
            hflip=torch.zeros(b, dtype=torch.bool),
            paste_valid=torch.ones((b, 3), dtype=torch.bool),
        )
        out = AffinePropagator()(padded, placement)
        src_boxes = padded.boxes[placement.source_idx]
        expected = src_boxes.clone()
        expected[0, :, [0, 2]] += 4.0
        expected[0, :, [1, 3]] += 5.0
        assert torch.allclose(out.boxes, expected)


class TestHflip:
    def test_hflip_mask_reflects(self) -> None:
        padded = _padded(b=2, k=3)
        placement = BatchedPlacement(
            source_idx=torch.tensor([1, 0]),
            translate=torch.zeros((2, 2)),
            scale=torch.ones(2),
            hflip=torch.tensor([True, False]),
            paste_valid=torch.ones((2, 3), dtype=torch.bool),
        )
        out = AffinePropagator()(padded, placement)
        assert padded.instance_masks is not None
        assert out.instance_masks is not None
        expected = padded.instance_masks[1, 0].flip(dims=[-1])
        assert torch.equal(out.instance_masks[0, 0], expected)

    def test_hflip_normals_x_sign_flips(self) -> None:
        padded = _padded(b=2, k=3, with_normals=True)
        placement = BatchedPlacement(
            source_idx=torch.tensor([1, 0]),
            translate=torch.zeros((2, 2)),
            scale=torch.ones(2),
            hflip=torch.tensor([True, False]),
            paste_valid=torch.ones((2, 3), dtype=torch.bool),
        )
        out = AffinePropagator()(padded, placement)
        assert padded.normals is not None
        assert out.normals is not None
        # Source normals x-channel reflected spatially AND negated on hflip row.
        expected_x = -padded.normals[1, 0].flip(dims=[-1])
        assert torch.allclose(out.normals[0, 0], expected_x, atol=1e-5)
        # Non-hflip row is unchanged.
        assert torch.allclose(out.normals[1, 0], padded.normals[0, 0], atol=1e-5)


class TestNearestMaskIntegrity:
    def test_masks_stay_bool_01(self) -> None:
        padded = _padded(b=2, k=3)
        placement = BatchedPlacement(
            source_idx=torch.tensor([1, 0]),
            translate=torch.tensor([[3.5, 2.5], [0.0, 0.0]]),
            scale=torch.full((2,), 1.25),
            hflip=torch.tensor([False, True]),
            paste_valid=torch.ones((2, 3), dtype=torch.bool),
        )
        out = AffinePropagator()(padded, placement)
        assert out.instance_masks is not None
        vals = out.instance_masks.unique()
        assert set(vals.tolist()).issubset({False, True})


class TestDepthValid:
    def test_depth_valid_false_outside_footprint(self) -> None:
        padded = _padded(b=2, k=3, with_depth=True)
        placement = BatchedPlacement(
            source_idx=torch.tensor([1, 0]),
            translate=torch.tensor([[H - 4.0, W - 4.0], [0.0, 0.0]]),
            scale=torch.full((2,), 0.25),
            hflip=torch.zeros(2, dtype=torch.bool),
            paste_valid=torch.ones((2, 3), dtype=torch.bool),
        )
        out = AffinePropagator()(padded, placement)
        assert out.depth_valid is not None
        # Origin is outside the translated footprint; should be invalid.
        assert not bool(out.depth_valid[0, 0, 0, 0].item())
