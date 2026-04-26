"""Tests for :class:`TileCompositor` (ADR-0008 C5)."""

from __future__ import annotations

from dataclasses import replace

import torch
from torchvision import tv_tensors

from segpaste._internal.gpu.tile_composite import TileCompositor, TileCompositorConfig
from segpaste.types import (
    BatchedDenseSample,
    CameraIntrinsics,
    DenseSample,
    InstanceMask,
    PaddedBatchedDenseSample,
    PaddingMask,
    PanopticMap,
    SemanticMap,
)

H = W = 64


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
        x2 = x1 + 10
        y2 = y1 + 10
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


def _padded(
    b: int = 2, k: int = 3, seed_base: int = 0, **kw: bool
) -> PaddedBatchedDenseSample:
    samples = [_sample(num_objects=2, seed=seed_base + i, **kw) for i in range(b)]
    return BatchedDenseSample.from_samples(samples).to_padded(max_instances=k)


def _paste_mask(b: int) -> torch.Tensor:
    m = torch.zeros(b, H, W, dtype=torch.bool)
    m[:, 10:40, 10:40] = True
    return m


class TestSingleTileEqualsMultiTile:
    def test_pixelwise_modalities_bitwise_match(self) -> None:
        target = _padded(
            b=2,
            k=3,
            seed_base=0,
            with_semantic=True,
            with_panoptic=True,
            with_depth=True,
            with_normals=True,
        )
        source = _padded(
            b=2,
            k=3,
            seed_base=10,
            with_semantic=True,
            with_panoptic=True,
            with_depth=True,
            with_normals=True,
        )
        pm = _paste_mask(2)

        single = TileCompositor(TileCompositorConfig(tile_size=max(H, W)))
        tiled = TileCompositor(TileCompositorConfig(tile_size=16))
        out_single = single(target, source, pm)
        out_tiled = tiled(target, source, pm)

        assert torch.equal(
            out_single.images.as_subclass(torch.Tensor),
            out_tiled.images.as_subclass(torch.Tensor),
        )
        assert (
            out_single.semantic_maps is not None and out_tiled.semantic_maps is not None
        )
        assert torch.equal(
            out_single.semantic_maps.as_subclass(torch.Tensor),
            out_tiled.semantic_maps.as_subclass(torch.Tensor),
        )
        assert (
            out_single.panoptic_maps is not None and out_tiled.panoptic_maps is not None
        )
        assert torch.equal(
            out_single.panoptic_maps.as_subclass(torch.Tensor),
            out_tiled.panoptic_maps.as_subclass(torch.Tensor),
        )
        assert out_single.depth is not None and out_tiled.depth is not None
        assert torch.equal(out_single.depth, out_tiled.depth)
        assert out_single.depth_valid is not None and out_tiled.depth_valid is not None
        assert torch.equal(out_single.depth_valid, out_tiled.depth_valid)
        assert out_single.normals is not None and out_tiled.normals is not None
        assert torch.equal(out_single.normals, out_tiled.normals)
        assert (
            out_single.instance_masks is not None
            and out_tiled.instance_masks is not None
        )
        assert torch.equal(out_single.instance_masks, out_tiled.instance_masks)


class TestWhereSemantics:
    def test_image_is_source_inside_mask_and_target_outside(self) -> None:
        target = _padded(b=2, k=3, seed_base=0)
        source = _padded(b=2, k=3, seed_base=10)
        pm = _paste_mask(2)
        comp = TileCompositor()
        out = comp(target, source, pm)
        out_t = out.images.as_subclass(torch.Tensor)
        tgt_t = target.images.as_subclass(torch.Tensor)
        src_t = source.images.as_subclass(torch.Tensor)
        m3 = pm.unsqueeze(1).expand_as(out_t)
        assert torch.equal(out_t[m3], src_t[m3])
        assert torch.equal(out_t[~m3], tgt_t[~m3])


class TestZTestDepth:
    def test_farther_source_rejected_inside_mask(self) -> None:
        target = _padded(b=1, k=2, seed_base=0, with_depth=True)
        source = _padded(b=1, k=2, seed_base=10, with_depth=True)
        assert target.depth is not None and source.depth is not None
        target_depth = torch.ones_like(target.depth) * 1.0
        source_depth = torch.ones_like(source.depth) * 5.0
        target = PaddedBatchedDenseSample(
            images=target.images,
            boxes=target.boxes,
            labels=target.labels,
            instance_valid=target.instance_valid,
            max_instances=target.max_instances,
            instance_masks=target.instance_masks,
            instance_ids=target.instance_ids,
            depth=target_depth,
            depth_valid=torch.ones_like(target_depth, dtype=torch.bool),
            camera_intrinsics=target.camera_intrinsics,
        )
        source = PaddedBatchedDenseSample(
            images=source.images,
            boxes=source.boxes,
            labels=source.labels,
            instance_valid=source.instance_valid,
            max_instances=source.max_instances,
            instance_masks=source.instance_masks,
            instance_ids=source.instance_ids,
            depth=source_depth,
            depth_valid=torch.ones_like(source_depth, dtype=torch.bool),
            camera_intrinsics=source.camera_intrinsics,
        )
        pm = torch.ones(1, H, W, dtype=torch.bool)
        out = TileCompositor()(target, source, pm)
        assert torch.equal(
            out.images.as_subclass(torch.Tensor),
            target.images.as_subclass(torch.Tensor),
        )


class TestSourcePaddingMask:
    def test_pad_pixels_keep_target_inside_pad_overlap(self) -> None:
        """Source-pad regions in the paste mask must not leak zeros into the output."""
        target = _padded(b=1, k=2, seed_base=0)
        source = _padded(b=1, k=2, seed_base=10)
        src_pad = torch.zeros(1, 1, H, W, dtype=torch.bool)
        src_pad[..., H // 2 :, W // 2 :] = True
        source_with_pad = replace(source, padding_mask=PaddingMask.from_tensor(src_pad))
        pm = torch.ones(1, H, W, dtype=torch.bool)
        out = TileCompositor()(target, source_with_pad, pm)
        out_t = out.images.as_subclass(torch.Tensor)
        tgt_t = target.images.as_subclass(torch.Tensor)
        src_t = source.images.as_subclass(torch.Tensor)
        m_pad3 = src_pad.squeeze(1).unsqueeze(1).expand_as(out_t)
        assert torch.equal(out_t[m_pad3], tgt_t[m_pad3])
        assert torch.equal(out_t[~m_pad3], src_t[~m_pad3])

    def test_pad_overlap_with_tiling_matches_single_tile(self) -> None:
        """Source-pad gating must be tile-invariant."""
        target = _padded(b=2, k=3, seed_base=0)
        source = _padded(b=2, k=3, seed_base=10)
        src_pad = torch.zeros(2, 1, H, W, dtype=torch.bool)
        src_pad[..., H // 2 :, :] = True
        source_with_pad = replace(source, padding_mask=PaddingMask.from_tensor(src_pad))
        pm = _paste_mask(2)
        single = TileCompositor(TileCompositorConfig(tile_size=max(H, W)))
        tiled = TileCompositor(TileCompositorConfig(tile_size=16))
        out_single = single(target, source_with_pad, pm)
        out_tiled = tiled(target, source_with_pad, pm)
        assert torch.equal(
            out_single.images.as_subclass(torch.Tensor),
            out_tiled.images.as_subclass(torch.Tensor),
        )


class TestSurvivorSubtraction:
    def test_target_masks_cleared_under_effective_mask(self) -> None:
        target = _padded(b=2, k=3, seed_base=0)
        source = _padded(b=2, k=3, seed_base=10)
        pm = _paste_mask(2)
        out = TileCompositor()(target, source, pm)
        assert out.instance_masks is not None
        assert target.instance_masks is not None
        m4 = pm.unsqueeze(1)
        assert not bool((out.instance_masks & m4).any())
        assert torch.equal(
            out.instance_masks & ~m4,
            target.instance_masks & ~m4,
        )
