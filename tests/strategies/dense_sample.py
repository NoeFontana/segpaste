"""Hypothesis strategies for DenseSample and its per-modality fields.

Design:

* Image size is capped at ``MAX_FUZZ_SIZE`` so Hypothesis shrink stays bounded
  on tensor payloads (ADR-0001 risk-mitigation note).
* Each modality has a standalone sub-strategy that returns the fields the
  modality adds to a DenseSample. ``dense_sample_strategy`` composes these
  based on the requested modality set.
* Strategies draw dtypes matching ADR-0001 §(iii): int64 labels, bool masks,
  int64 semantic/panoptic maps, float32 depth / normals, bool depth-valid.
"""

from typing import Any

import torch
from hypothesis import strategies as st
from torchvision import tv_tensors

from segpaste.types import (
    DenseSample,
    InstanceMask,
    Modality,
    PanopticMap,
    SemanticMap,
)

MAX_FUZZ_SIZE = 64
_MIN_FUZZ_SIZE = 8
_MAX_OBJECTS = 5
_MAX_CLASSES = 8


@st.composite
def image_strategy(
    draw: st.DrawFn,
    min_size: int = _MIN_FUZZ_SIZE,
    max_size: int = MAX_FUZZ_SIZE,
) -> tv_tensors.Image:
    """Draw a ``[3, H, W]`` float32 image in [0, 1]."""
    h = draw(st.integers(min_value=min_size, max_value=max_size))
    w = draw(st.integers(min_value=min_size, max_value=max_size))
    data = torch.rand(3, h, w, generator=_seeded_generator(draw))
    return tv_tensors.Image(data)


def _seeded_generator(draw: st.DrawFn) -> torch.Generator:
    """Hypothesis-controlled seed for torch RNG inside a strategy."""
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


@st.composite
def instance_fields_strategy(
    draw: st.DrawFn,
    h: int,
    w: int,
    max_objects: int = _MAX_OBJECTS,
    max_classes: int = _MAX_CLASSES,
) -> dict[str, Any]:
    """Draw ``{"boxes", "labels", "instance_masks"}`` consistent with ``(h, w)``.

    Boxes are pixel-integer xyxy coordinates; masks are a random sub-rectangle
    inside the box (bool dtype).
    """
    n = draw(st.integers(min_value=0, max_value=max_objects))
    if n == 0:
        return {
            "boxes": tv_tensors.BoundingBoxes(  # pyright: ignore[reportCallIssue]
                torch.zeros((0, 4), dtype=torch.float32),
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=(h, w),
            ),
            "labels": torch.zeros((0,), dtype=torch.int64),
            "instance_ids": torch.zeros((0,), dtype=torch.int32),
            "instance_masks": InstanceMask(torch.zeros((0, h, w), dtype=torch.bool)),
        }

    boxes_list: list[list[int]] = []
    labels_list: list[int] = []
    masks = torch.zeros((n, h, w), dtype=torch.bool)

    for i in range(n):
        mx1 = draw(st.integers(min_value=0, max_value=w - 2))
        my1 = draw(st.integers(min_value=0, max_value=h - 2))
        mx2 = draw(st.integers(min_value=mx1 + 1, max_value=w - 1))
        my2 = draw(st.integers(min_value=my1 + 1, max_value=h - 1))
        masks[i, my1:my2, mx1:mx2] = True
        # Boxes are the tight bbox of the mask so that every invariant
        # predicate starts from a bbox-consistent sample.
        boxes_list.append([mx1, my1, mx2, my2])
        labels_list.append(draw(st.integers(min_value=1, max_value=max_classes)))

    return {
        "boxes": tv_tensors.BoundingBoxes(  # pyright: ignore[reportCallIssue]
            torch.tensor(boxes_list, dtype=torch.float32),
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=(h, w),
        ),
        "labels": torch.tensor(labels_list, dtype=torch.int64),
        "instance_ids": torch.arange(n, dtype=torch.int32),
        "instance_masks": InstanceMask(masks),
    }


@st.composite
def semantic_map_strategy(
    draw: st.DrawFn,
    h: int,
    w: int,
    max_classes: int = _MAX_CLASSES,
) -> SemanticMap:
    """Draw an ``[H, W]`` int64 semantic map.

    Class labels are in ``[0, max_classes)``; ``255`` is reserved as the ignore
    label and appears only if Hypothesis chooses to inject it (controlled below).
    """
    gen = _seeded_generator(draw)
    data = torch.randint(0, max_classes, (h, w), dtype=torch.int64, generator=gen)
    # Inject a small ignore region with low probability so ignore-preservation
    # invariants have cases to check.
    if draw(st.booleans()):
        mask = torch.rand(h, w, generator=gen) < 0.05
        data[mask] = 255
    return SemanticMap(data)


@st.composite
def panoptic_map_strategy(
    draw: st.DrawFn,
    h: int,
    w: int,
    max_classes: int = _MAX_CLASSES,
) -> PanopticMap:
    """Draw an ``[H, W]`` int64 panoptic id map.

    For the skeleton, ids are arbitrary non-negative integers < ``max_classes``.
    Real schema-aware strategies will refine this in P1 alongside the
    panoptic composite.
    """
    gen = _seeded_generator(draw)
    data = torch.randint(0, max_classes, (h, w), dtype=torch.int64, generator=gen)
    return PanopticMap(data)


@st.composite
def depth_fields_strategy(
    draw: st.DrawFn,
    h: int,
    w: int,
) -> dict[str, torch.Tensor]:
    """Draw ``{"depth", "depth_valid"}`` with matching ``[1, H, W]`` shape."""
    gen = _seeded_generator(draw)
    depth = torch.rand(1, h, w, generator=gen) * 10.0
    depth_valid = torch.rand(1, h, w, generator=gen) > 0.1
    return {"depth": depth, "depth_valid": depth_valid}


@st.composite
def normals_strategy(draw: st.DrawFn, h: int, w: int) -> torch.Tensor:
    """Draw a ``[3, H, W]`` float32 unit-norm normals tensor."""
    gen = _seeded_generator(draw)
    raw = torch.randn(3, h, w, generator=gen)
    norm = raw.norm(dim=0, keepdim=True).clamp(min=1e-6)
    return raw / norm


@st.composite
def dense_sample_strategy(
    draw: st.DrawFn,
    modalities: set[Modality],
) -> DenseSample:
    """Draw a :class:`DenseSample` carrying the requested modalities.

    ``Modality.IMAGE`` is always active; other members add the corresponding
    fields. Instance fields are always present (at least as empty tensors)
    because :class:`DenseSample` requires ``boxes`` and ``labels``; the
    ``INSTANCE`` modality flag controls whether ``instance_masks`` is
    populated.
    """
    image = draw(image_strategy())
    h, w = image.shape[-2:]

    fields: dict[str, Any] = {"image": image}

    if Modality.INSTANCE in modalities:
        fields.update(draw(instance_fields_strategy(h, w)))
    else:
        fields["boxes"] = tv_tensors.BoundingBoxes(  # pyright: ignore[reportCallIssue]
            torch.zeros((0, 4), dtype=torch.float32),
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=(h, w),
        )
        fields["labels"] = torch.zeros((0,), dtype=torch.int64)

    if Modality.SEMANTIC in modalities:
        fields["semantic_map"] = draw(semantic_map_strategy(h, w))
    if Modality.PANOPTIC in modalities:
        fields["panoptic_map"] = draw(panoptic_map_strategy(h, w))
    if Modality.DEPTH in modalities:
        fields.update(draw(depth_fields_strategy(h, w)))
    if Modality.NORMALS in modalities:
        fields["normals"] = draw(normals_strategy(h, w))

    return DenseSample(**fields)
