"""Shape and dtype invariants for :class:`BatchCopyPaste` (ADR-0008 C7)."""

from __future__ import annotations

import torch
from torchvision import tv_tensors

from segpaste import BatchCopyPaste, PaddedBatchedDenseSample
from segpaste._internal.gpu.tile_composite import TileCompositorConfig
from segpaste.augmentation.batch_copy_paste import BatchCopyPasteConfig
from segpaste.types import (
    BatchedDenseSample,
    CameraIntrinsics,
    DenseSample,
    InstanceMask,
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


def _padded(b: int = 2, k: int = 3, **kw: bool) -> PaddedBatchedDenseSample:
    samples = [_sample(num_objects=2, seed=i, **kw) for i in range(b)]
    return BatchedDenseSample.from_samples(samples).to_padded(max_instances=k)


class TestShape:
    def test_instance_only_batch(self) -> None:
        padded = _padded(b=4, k=5)
        out = BatchCopyPaste()(padded, torch.Generator().manual_seed(0))
        assert isinstance(out, PaddedBatchedDenseSample)
        assert out.images.shape == (4, 3, H, W)
        assert out.boxes.shape == (4, 5, 4)
        assert out.labels.shape == (4, 5)
        assert out.instance_valid.shape == (4, 5)
        assert out.instance_masks is not None and out.instance_masks.shape == (
            4,
            5,
            H,
            W,
        )
        assert out.instance_masks.dtype == torch.bool

    def test_all_modalities(self) -> None:
        padded = _padded(
            b=2,
            k=3,
            with_semantic=True,
            with_panoptic=True,
            with_depth=True,
            with_normals=True,
        )
        out = BatchCopyPaste()(padded, torch.Generator().manual_seed(0))
        assert out.semantic_maps is not None and out.semantic_maps.shape == (2, H, W)
        assert out.panoptic_maps is not None and out.panoptic_maps.shape == (2, H, W)
        assert out.depth is not None and out.depth.shape == (2, 1, H, W)
        assert out.depth_valid is not None and out.depth_valid.dtype == torch.bool
        assert out.normals is not None and out.normals.shape == (2, 3, H, W)

    def test_b_one_is_identity(self) -> None:
        padded = _padded(b=1, k=3)
        out = BatchCopyPaste()(padded, torch.Generator().manual_seed(0))
        assert out.instance_valid.sum() == 0 or torch.equal(
            out.images.as_subclass(torch.Tensor),
            padded.images.as_subclass(torch.Tensor),
        )

    def test_b_zero_returns_input(self) -> None:
        empty = BatchedDenseSample.from_samples([]).to_padded(max_instances=3)
        out = BatchCopyPaste()(empty, torch.Generator().manual_seed(0))
        assert out.batch_size == 0


class TestConfigValidation:
    def test_tile_size_positive(self) -> None:
        cfg = BatchCopyPasteConfig(composite=TileCompositorConfig(tile_size=64))
        assert cfg.composite.tile_size == 64

    def test_tile_size_zero_rejected(self) -> None:
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            TileCompositorConfig(tile_size=0)
