"""Deterministic synthetic ``DenseSample`` builders."""

from collections.abc import Callable
from typing import Any

import torch
from torchvision import tv_tensors

from segpaste.types import (
    DenseSample,
    InstanceMask,
    PanopticMap,
    SemanticMap,
)


def _image(h: int, w: int, seed: int) -> tv_tensors.Image:
    gen = torch.Generator().manual_seed(seed)
    return tv_tensors.Image(torch.rand(3, h, w, generator=gen))


def _empty_instance(h: int, w: int) -> dict[str, Any]:
    return {
        "boxes": tv_tensors.BoundingBoxes(  # pyright: ignore[reportCallIssue]
            torch.zeros((0, 4), dtype=torch.float32),
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=(h, w),
        ),
        "labels": torch.zeros((0,), dtype=torch.int64),
    }


def build_two_overlapping_things() -> DenseSample:
    """Two instance masks of different classes with a known overlap region."""
    h, w = 32, 32
    masks = torch.zeros((2, h, w), dtype=torch.bool)
    masks[0, 4:20, 4:20] = True  # box 1: [4,4,20,20]
    masks[1, 12:28, 12:28] = True  # box 2: [12,12,28,28], overlaps in [12,12,20,20]
    boxes = torch.tensor(
        [[4.0, 4.0, 20.0, 20.0], [12.0, 12.0, 28.0, 28.0]], dtype=torch.float32
    )
    labels = torch.tensor([1, 2], dtype=torch.int64)
    return DenseSample(
        image=_image(h, w, seed=0),
        boxes=tv_tensors.BoundingBoxes(  # pyright: ignore[reportCallIssue]
            boxes, format=tv_tensors.BoundingBoxFormat.XYXY, canvas_size=(h, w)
        ),
        labels=labels,
        instance_masks=InstanceMask(masks),
    )


def build_empty_instance() -> DenseSample:
    """Instance modality with zero objects — the boundary case."""
    h, w = 16, 16
    return DenseSample(
        image=_image(h, w, seed=1),
        **_empty_instance(h, w),
        instance_masks=InstanceMask(torch.zeros((0, h, w), dtype=torch.bool)),
    )


def build_semantic_with_ignore() -> DenseSample:
    """Semantic map with a contiguous ignore (255) region in the upper-left."""
    h, w = 32, 32
    sem = torch.full((h, w), 1, dtype=torch.int64)
    sem[:8, :8] = 255  # ignore region
    sem[16:, 16:] = 2  # second class
    return DenseSample(
        image=_image(h, w, seed=2),
        **_empty_instance(h, w),
        semantic_map=SemanticMap(sem),
    )


def build_panoptic_stuff_and_things() -> DenseSample:
    """Panoptic + semantic + instance — two things on a stuff background.

    Class 0 is stuff (``sky``), class 1 and class 2 are things.
    Panoptic ids follow ADR-0001: ``z(p) == 0`` iff stuff pixel, otherwise
    a unique id per instance.
    """
    h, w = 32, 32
    masks = torch.zeros((2, h, w), dtype=torch.bool)
    masks[0, 8:16, 4:12] = True
    masks[1, 18:28, 20:30] = True
    boxes = torch.tensor(
        [[4.0, 8.0, 12.0, 16.0], [20.0, 18.0, 30.0, 28.0]], dtype=torch.float32
    )
    labels = torch.tensor([1, 2], dtype=torch.int64)

    sem = torch.zeros((h, w), dtype=torch.int64)  # class 0 = stuff (sky)
    sem[masks[0]] = 1
    sem[masks[1]] = 2

    pan = torch.zeros((h, w), dtype=torch.int64)
    pan[masks[0]] = 1
    pan[masks[1]] = 2

    return DenseSample(
        image=_image(h, w, seed=3),
        boxes=tv_tensors.BoundingBoxes(  # pyright: ignore[reportCallIssue]
            boxes, format=tv_tensors.BoundingBoxFormat.XYXY, canvas_size=(h, w)
        ),
        labels=labels,
        instance_masks=InstanceMask(masks),
        semantic_map=SemanticMap(sem),
        panoptic_map=PanopticMap(pan),
    )


def build_depth_two_planes() -> DenseSample:
    """Depth scene with two fronto-parallel planes at known z values.

    Left half at depth ``2.0``, right half at depth ``5.0``; everything valid.
    Useful for checking the ``min(d_src, d_tgt)`` composite rule.
    """
    h, w = 32, 32
    depth = torch.full((1, h, w), 5.0, dtype=torch.float32)
    depth[:, :, : w // 2] = 2.0
    valid = torch.ones((1, h, w), dtype=torch.bool)
    return DenseSample(
        image=_image(h, w, seed=4),
        **_empty_instance(h, w),
        depth=depth,
        depth_valid=valid,
    )


def build_normals_sphere_patch() -> DenseSample:
    """Analytic unit-norm normals: one half tilted ``+x``, the other ``-x``."""
    h, w = 32, 32
    normals = torch.zeros((3, h, w), dtype=torch.float32)
    normals[2] = 1.0  # z = 1 baseline (camera-forward)
    normals[0, :, : w // 2] = 1.0  # x = +1 left half
    normals[0, :, w // 2 :] = -1.0  # x = -1 right half
    # Renormalize to enforce unit norm on every pixel
    normals = normals / normals.norm(dim=0, keepdim=True).clamp(min=1e-6)
    return DenseSample(
        image=_image(h, w, seed=5),
        **_empty_instance(h, w),
        normals=normals,
    )


BUILDERS: dict[str, Callable[[], DenseSample]] = {
    "two_overlapping_things": build_two_overlapping_things,
    "empty_instance": build_empty_instance,
    "semantic_with_ignore": build_semantic_with_ignore,
    "panoptic_stuff_and_things": build_panoptic_stuff_and_things,
    "depth_two_planes": build_depth_two_planes,
    "normals_sphere_patch": build_normals_sphere_patch,
}
