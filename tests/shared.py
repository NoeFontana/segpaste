from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import torch
from torchvision import tv_tensors
from torchvision.transforms import v2

from segpaste.augmentation import make_large_scale_jittering
from segpaste.augmentation.lsj import SanitizeBoundingBoxes
from segpaste.integrations import labels_getter
from segpaste.types import (
    DenseSample,
    InstanceMask,
    PanopticMap,
    PanopticSchema,
    SemanticMap,
)


@dataclass
class FakePanopticSchema:
    """Lightweight :class:`PanopticSchema` stand-in for tests.

    Avoids importing the production schema type's full validation surface
    when a test just needs the protocol fields (``classes``, ``ignore_index``,
    ``max_instances_per_image``).
    """

    classes: Mapping[int, Literal["thing", "stuff"]]
    ignore_index: int
    max_instances_per_image: int


def make_disjoint_panoptic_sample(seed: int = 0) -> DenseSample:
    """Two non-overlapping things with mutually-consistent maps + masks.

    Used by tests that need a sample satisfying the panoptic pixel-bijection
    and same-class-overlap invariants — properties the random Hypothesis
    strategy does not guarantee.
    """
    g = torch.Generator().manual_seed(seed)
    h, w = 24, 24
    image = tv_tensors.Image(torch.rand(3, h, w, generator=g, dtype=torch.float32))
    masks = torch.zeros((2, h, w), dtype=torch.bool)
    masks[0, 2:10, 2:10] = True
    masks[1, 12:20, 12:20] = True
    labels = torch.tensor([1, 2], dtype=torch.int64)
    boxes = tv_tensors.BoundingBoxes(  # pyright: ignore[reportCallIssue]
        torch.tensor(
            [[2.0, 2.0, 10.0, 10.0], [12.0, 12.0, 20.0, 20.0]], dtype=torch.float32
        ),
        format=tv_tensors.BoundingBoxFormat.XYXY,
        canvas_size=(h, w),
    )
    sem = torch.zeros((h, w), dtype=torch.int64)
    sem[masks[0]] = 1
    sem[masks[1]] = 2
    pan = torch.zeros((h, w), dtype=torch.int64)
    pan[masks[0]] = 1
    pan[masks[1]] = 2
    return DenseSample(
        image=image,
        boxes=boxes,
        labels=labels,
        instance_ids=torch.arange(2, dtype=torch.int32),
        instance_masks=InstanceMask(masks),
        semantic_map=SemanticMap(sem),
        panoptic_map=PanopticMap(pan),
    )


def make_thing_stuff_schema() -> PanopticSchema:
    """Schema for :func:`make_disjoint_panoptic_sample` (class 0 stuff, 1+ thing)."""
    return FakePanopticSchema(
        classes={0: "stuff", 1: "thing", 2: "thing"},
        ignore_index=255,
        max_instances_per_image=256,
    )


def generate_scale_jitter_transform_strategy(
    min_scale: float = 0.1, max_scale: float = 2.0
) -> v2.Transform:
    return v2.Compose(
        [
            v2.ToImage(),
            v2.RandomHorizontalFlip(),
            make_large_scale_jittering(
                output_size=(256, 256), min_scale=min_scale, max_scale=max_scale
            ),
            v2.ClampBoundingBoxes(),
            SanitizeBoundingBoxes(labels_getter=labels_getter),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )


def generate_resize_transform_strategy() -> v2.Transform:
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(size=(256, 256)),
            v2.RandomHorizontalFlip(),
            v2.ClampBoundingBoxes(),
            SanitizeBoundingBoxes(labels_getter=labels_getter),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
