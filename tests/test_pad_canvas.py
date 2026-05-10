"""Tests for :func:`pad_canvas_to_multiple` (A2)."""

from __future__ import annotations

import pytest
import torch
from torchvision import tv_tensors

from segpaste._internal.gpu.pad_canvas import pad_canvas_to_multiple
from segpaste.types import (
    BatchedDenseSample,
    DenseSample,
    InstanceMask,
    PaddedBatchedDenseSample,
    PaddingMask,
    PanopticMap,
    SemanticMap,
)


def _sample(
    h: int,
    w: int,
    seed: int = 0,
    *,
    with_semantic: bool = False,
    with_panoptic: bool = False,
    with_depth: bool = False,
    with_normals: bool = False,
    with_padding_mask: bool = False,
) -> DenseSample:
    gen = torch.Generator().manual_seed(seed)
    masks = torch.zeros(1, h, w, dtype=torch.bool)
    masks[0, : h // 2, : w // 2] = True
    fields: dict[str, object] = {
        "image": tv_tensors.Image(torch.rand(3, h, w, generator=gen)),
        "boxes": tv_tensors.BoundingBoxes(  # pyright: ignore[reportCallIssue]
            torch.tensor([[0.0, 0.0, w / 2.0, h / 2.0]], dtype=torch.float32),
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=(h, w),
        ),
        "labels": torch.tensor([1], dtype=torch.int64),
        "instance_ids": torch.tensor([0], dtype=torch.int32),
        "instance_masks": InstanceMask(masks),
    }
    if with_semantic:
        sem = torch.zeros((h, w), dtype=torch.int64)
        sem[: h // 2, : w // 2] = 1
        fields["semantic_map"] = SemanticMap(sem)
    if with_panoptic:
        pan = torch.zeros((h, w), dtype=torch.int64)
        pan[: h // 2, : w // 2] = 1
        fields["panoptic_map"] = PanopticMap(pan)
    if with_depth:
        fields["depth"] = torch.rand(1, h, w, generator=gen) * 10.0
        fields["depth_valid"] = torch.ones(1, h, w, dtype=torch.bool)
    if with_normals:
        raw = torch.randn(3, h, w, generator=gen)
        fields["normals"] = raw / raw.norm(dim=0, keepdim=True).clamp(min=1e-6)
    if with_padding_mask:
        fields["padding_mask"] = PaddingMask.from_tensor(
            torch.zeros((1, h, w), dtype=torch.bool)
        )
    return DenseSample(**fields)  # pyright: ignore[reportArgumentType]


def _padded(b: int, h: int, w: int, **kwargs: bool) -> PaddedBatchedDenseSample:
    samples = [_sample(h=h, w=w, seed=i, **kwargs) for i in range(b)]
    return BatchedDenseSample.from_samples(samples).to_padded(max_instances=1)


class TestDimensions:
    def test_pads_to_multiple(self) -> None:
        padded = _padded(b=2, h=30, w=22)
        out = pad_canvas_to_multiple(padded, p=14, ignore_index=255)
        assert out.images.shape[-2:] == (42, 28)  # next multiples of 14

    def test_already_divisible_returns_same_object(self) -> None:
        padded = _padded(b=2, h=28, w=28)
        out = pad_canvas_to_multiple(padded, p=14, ignore_index=255)
        assert out is padded

    def test_p_one_is_identity(self) -> None:
        padded = _padded(b=2, h=15, w=15)
        out = pad_canvas_to_multiple(padded, p=1, ignore_index=255)
        assert out is padded

    def test_invalid_p_rejected(self) -> None:
        padded = _padded(b=2, h=15, w=15)
        with pytest.raises(ValueError):
            pad_canvas_to_multiple(padded, p=0, ignore_index=255)


class TestImageReflectPad:
    def test_image_reflected_on_right_band(self) -> None:
        padded = _padded(b=1, h=30, w=22)
        out = pad_canvas_to_multiple(padded, p=14, ignore_index=255)
        # Right band [22, 28) reflects from columns [21, 15) of the source.
        img_in = padded.images.as_subclass(torch.Tensor)
        img_out = out.images.as_subclass(torch.Tensor)
        for offset in range(1, 7):
            assert torch.allclose(
                img_out[:, :, : padded.images.shape[-2], 22 + offset - 1],
                img_in[:, :, :, 22 - 1 - offset],
            )


class TestMaskAndMapPads:
    def test_instance_masks_zero_pad(self) -> None:
        padded = _padded(b=2, h=30, w=22)
        out = pad_canvas_to_multiple(padded, p=14, ignore_index=255)
        assert out.instance_masks is not None
        # Right band columns [22, 28) are False on instance masks.
        assert not bool(out.instance_masks[..., 22:].any())
        # Bottom band rows [30, 42) are False on instance masks.
        assert not bool(out.instance_masks[..., 30:, :].any())

    def test_semantic_panoptic_filled_with_ignore_index(self) -> None:
        padded = _padded(b=1, h=30, w=22, with_semantic=True, with_panoptic=True)
        out = pad_canvas_to_multiple(padded, p=14, ignore_index=255)
        assert out.semantic_maps is not None
        assert out.panoptic_maps is not None
        sem = out.semantic_maps.as_subclass(torch.Tensor)
        pan = out.panoptic_maps.as_subclass(torch.Tensor)
        assert bool((sem[..., :, 22:] == 255).all())
        assert bool((sem[..., 30:, :] == 255).all())
        assert bool((pan[..., :, 22:] == 255).all())
        assert bool((pan[..., 30:, :] == 255).all())

    def test_custom_ignore_index_propagates(self) -> None:
        padded = _padded(b=1, h=30, w=22, with_semantic=True)
        out = pad_canvas_to_multiple(padded, p=14, ignore_index=42)
        assert out.semantic_maps is not None
        sem = out.semantic_maps.as_subclass(torch.Tensor)
        assert bool((sem[..., :, 22:] == 42).all())

    def test_depth_zero_pad_and_depth_valid_false(self) -> None:
        padded = _padded(b=1, h=30, w=22, with_depth=True)
        out = pad_canvas_to_multiple(padded, p=14, ignore_index=255)
        assert out.depth is not None and out.depth_valid is not None
        assert bool((out.depth[..., :, 22:] == 0.0).all())
        assert not bool(out.depth_valid[..., :, 22:].any())

    def test_normals_zero_pad(self) -> None:
        padded = _padded(b=1, h=30, w=22, with_normals=True)
        out = pad_canvas_to_multiple(padded, p=14, ignore_index=255)
        assert out.normals is not None
        assert bool((out.normals[..., :, 22:] == 0.0).all())


class TestPaddingMask:
    def test_padding_mask_created_when_input_none(self) -> None:
        padded = _padded(b=1, h=30, w=22)
        assert padded.padding_mask is None
        out = pad_canvas_to_multiple(padded, p=14, ignore_index=255)
        assert out.padding_mask is not None
        pm = out.padding_mask.as_subclass(torch.Tensor)
        assert pm.shape == (1, 1, 42, 28)
        # Original area is False (not pad), new band is True.
        assert not bool(pm[..., :30, :22].any())
        assert bool(pm[..., :, 22:].all())
        assert bool(pm[..., 30:, :].all())

    def test_existing_padding_mask_extended(self) -> None:
        padded = _padded(b=1, h=30, w=22, with_padding_mask=True)
        out = pad_canvas_to_multiple(padded, p=14, ignore_index=255)
        assert out.padding_mask is not None
        pm = out.padding_mask.as_subclass(torch.Tensor)
        assert pm.shape == (1, 1, 42, 28)
        assert bool(pm[..., :, 22:].all())


class TestForwardedFields:
    def test_boxes_unchanged(self) -> None:
        padded = _padded(b=2, h=30, w=22)
        out = pad_canvas_to_multiple(padded, p=14, ignore_index=255)
        assert torch.equal(out.boxes, padded.boxes)

    def test_instance_valid_unchanged(self) -> None:
        padded = _padded(b=2, h=30, w=22)
        out = pad_canvas_to_multiple(padded, p=14, ignore_index=255)
        assert torch.equal(out.instance_valid, padded.instance_valid)
