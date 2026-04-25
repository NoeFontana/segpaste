"""Deterministic synthetic-source factory for the viz ritual.

Produces instance-modality :class:`DenseSample` instances modeled on
``scripts/compile_explain.py::_sample`` — image plus two non-overlapping
square instances with consistent boxes/labels/masks.
"""

from __future__ import annotations

import torch
from torchvision import tv_tensors

from segpaste.types import DenseSample, InstanceMask

_IMAGE_SIZE = 64
_BOX_SIZE = 16
_NUM_INSTANCES = 2


def make_synthetic_samples(
    seed: int, count: int, image_size: int = _IMAGE_SIZE
) -> list[DenseSample]:
    """Return *count* deterministic :class:`DenseSample` instances.

    Each sample carries a random image and two non-overlapping square
    instances at seed-derived positions. Active modalities: image,
    instance.
    """
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")
    if image_size < 2 * _BOX_SIZE + 4:
        raise ValueError(
            f"image_size={image_size} too small for two {_BOX_SIZE}px boxes"
        )

    return [_make_one(seed=seed + i, image_size=image_size) for i in range(count)]


def _make_one(seed: int, image_size: int) -> DenseSample:
    g = torch.Generator().manual_seed(seed)
    h = w = image_size

    image = tv_tensors.Image(torch.rand(3, h, w, generator=g, dtype=torch.float32))

    free_range = image_size - _BOX_SIZE
    y0a = int(torch.randint(0, free_range // 2, (1,), generator=g).item())
    x0a = int(torch.randint(0, free_range // 2, (1,), generator=g).item())
    y0b = int(torch.randint(free_range // 2 + 2, free_range, (1,), generator=g).item())
    x0b = int(torch.randint(free_range // 2 + 2, free_range, (1,), generator=g).item())
    corners = ((y0a, x0a), (y0b, x0b))

    masks = torch.zeros((_NUM_INSTANCES, h, w), dtype=torch.bool)
    boxes_xyxy: list[list[float]] = []
    for idx, (y0, x0) in enumerate(corners):
        y1, x1 = y0 + _BOX_SIZE, x0 + _BOX_SIZE
        masks[idx, y0:y1, x0:x1] = True
        boxes_xyxy.append([float(x0), float(y0), float(x1), float(y1)])

    boxes = tv_tensors.BoundingBoxes(  # pyright: ignore[reportCallIssue]
        torch.tensor(boxes_xyxy, dtype=torch.float32),
        format=tv_tensors.BoundingBoxFormat.XYXY,
        canvas_size=(h, w),
    )
    labels = torch.tensor([1, 2], dtype=torch.int64)
    return DenseSample(
        image=image,
        boxes=boxes,
        labels=labels,
        instance_ids=torch.arange(_NUM_INSTANCES, dtype=torch.int32),
        instance_masks=InstanceMask(masks),
    )
