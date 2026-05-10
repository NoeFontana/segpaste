"""Bitwise-equivalence gate for ``BatchCopyPaste`` default config.

Pins the per-tensor output of ``BatchCopyPaste(default_config).forward(...)``
against a committed snapshot. The snapshot is generated from the v0.3.0
forward output and carried across the A1 InstanceBank refactor (ADR-0011)
to guarantee that ``IntraBatchSource`` (the default) does not perturb
RNG draw order or any tensor value relative to v0.3.0.

Regenerate the snapshot with::

    SEGPASTE_REGEN_BITWISE=1 uv run pytest tests/test_batch_copy_paste_bitwise.py

The test then writes ``tests/fixtures/batch_copy_paste_v0_3_0.pt`` and
exits with no assertion. Subsequent runs load and compare.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from torchvision import tv_tensors

from segpaste import BatchCopyPaste, PaddedBatchedDenseSample
from segpaste.types import (
    BatchedDenseSample,
    DenseSample,
    InstanceMask,
    SemanticMap,
)

SNAPSHOT_PATH = Path(__file__).parent / "fixtures" / "batch_copy_paste_v0_3_0.pt"
H = W = 32
B = 4
K = 6


def _sample(seed: int) -> DenseSample:
    gen = torch.Generator().manual_seed(seed * 2654435761 & 0xFFFFFFFF)
    n = int(torch.randint(2, 5, (1,), generator=gen).item())
    image = tv_tensors.Image(torch.rand(3, H, W, generator=gen))
    masks = torch.zeros(n, H, W, dtype=torch.bool)
    boxes: list[list[int]] = []
    for i in range(n):
        side = int(torch.randint(6, 12, (1,), generator=gen).item())
        x1 = int(torch.randint(0, W - side, (1,), generator=gen).item())
        y1 = int(torch.randint(0, H - side, (1,), generator=gen).item())
        masks[i, y1 : y1 + side, x1 : x1 + side] = True
        boxes.append([x1, y1, x1 + side, y1 + side])
    semantic = SemanticMap(
        torch.randint(0, 5, (H, W), generator=gen, dtype=torch.int64)
    )
    return DenseSample(
        image=image,
        boxes=tv_tensors.BoundingBoxes(  # pyright: ignore[reportCallIssue]
            torch.tensor(boxes, dtype=torch.float32),
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=(H, W),
        ),
        labels=torch.randint(1, 5, (n,), generator=gen, dtype=torch.int64),
        instance_ids=torch.arange(n, dtype=torch.int32),
        instance_masks=InstanceMask(masks),
        semantic_map=semantic,
    )


def _padded() -> PaddedBatchedDenseSample:
    samples = [_sample(seed=i) for i in range(B)]
    return BatchedDenseSample.from_samples(samples).to_padded(max_instances=K)


def _forward() -> PaddedBatchedDenseSample:
    padded = _padded()
    module = BatchCopyPaste()
    gen = torch.Generator().manual_seed(0xC0FFEE)
    with torch.no_grad():
        return module(padded, generator=gen)


_TENSOR_FIELDS = (
    "images",
    "boxes",
    "labels",
    "instance_valid",
    "instance_masks",
    "instance_ids",
    "semantic_maps",
    "panoptic_maps",
    "depth",
    "depth_valid",
    "normals",
    "padding_mask",
    "camera_intrinsics",
)


def _to_flat_dict(
    out: PaddedBatchedDenseSample,
) -> tuple[int, dict[str, torch.Tensor | None]]:
    tensors: dict[str, torch.Tensor | None] = {
        "images": out.images.as_subclass(torch.Tensor).clone(),
        "boxes": out.boxes.clone(),
        "labels": out.labels.clone(),
        "instance_valid": out.instance_valid.clone(),
        "instance_masks": (
            out.instance_masks.clone() if out.instance_masks is not None else None
        ),
        "instance_ids": (
            out.instance_ids.clone() if out.instance_ids is not None else None
        ),
        "semantic_maps": (
            out.semantic_maps.as_subclass(torch.Tensor).clone()
            if out.semantic_maps is not None
            else None
        ),
        "panoptic_maps": (
            out.panoptic_maps.as_subclass(torch.Tensor).clone()
            if out.panoptic_maps is not None
            else None
        ),
        "depth": out.depth.clone() if out.depth is not None else None,
        "depth_valid": (
            out.depth_valid.clone() if out.depth_valid is not None else None
        ),
        "normals": out.normals.clone() if out.normals is not None else None,
        "padding_mask": (
            out.padding_mask.as_subclass(torch.Tensor).clone()
            if out.padding_mask is not None
            else None
        ),
        "camera_intrinsics": (
            out.camera_intrinsics.clone() if out.camera_intrinsics is not None else None
        ),
    }
    return out.max_instances, tensors


def _assert_equal_field(
    name: str, got: torch.Tensor | None, want: torch.Tensor | None
) -> None:
    assert (got is None) == (want is None), f"{name}: presence mismatch"
    if got is None or want is None:
        return
    assert got.dtype == want.dtype, f"{name}: dtype {got.dtype} != {want.dtype}"
    assert got.shape == want.shape, f"{name}: shape {got.shape} != {want.shape}"
    assert torch.equal(got, want), f"{name}: tensor values diverge"


def test_default_forward_matches_v0_3_0_snapshot() -> None:
    out = _forward()
    max_instances, tensors = _to_flat_dict(out)
    flat: dict[str, torch.Tensor | None | int] = {
        "max_instances": max_instances,
        **tensors,
    }

    if os.environ.get("SEGPASTE_REGEN_BITWISE") == "1":
        SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(flat, SNAPSHOT_PATH)
        pytest.skip(f"regenerated snapshot at {SNAPSHOT_PATH}")

    if not SNAPSHOT_PATH.is_file():
        pytest.skip(
            f"snapshot missing at {SNAPSHOT_PATH}; "
            "regenerate with SEGPASTE_REGEN_BITWISE=1"
        )

    snap = torch.load(SNAPSHOT_PATH, weights_only=True)
    assert max_instances == snap["max_instances"]
    for name in _TENSOR_FIELDS:
        _assert_equal_field(name, tensors[name], snap[name])
