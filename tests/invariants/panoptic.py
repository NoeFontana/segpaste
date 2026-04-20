"""Panoptic-modality invariants per ADR-0001 §(ii)."""

import torch

from segpaste.types import DenseSample, PanopticSchema
from tests.invariants import require


def assert_panoptic_thing_stuff_consistent(
    sample: DenseSample, schema: PanopticSchema
) -> None:
    """``z(p) == 0`` iff ``s(p)`` is a stuff class."""
    z = torch.as_tensor(require(sample.panoptic_map, "sample must carry panoptic_map"))
    s = torch.as_tensor(require(sample.semantic_map, "sample must carry semantic_map"))

    stuff_classes = torch.tensor(
        [c for c, kind in schema.classes.items() if kind == "stuff"],
        dtype=s.dtype,
    )
    if stuff_classes.numel() == 0:
        is_stuff = torch.zeros_like(s, dtype=torch.bool)
    else:
        is_stuff = torch.isin(s, stuff_classes)

    ignore = s == schema.ignore_index
    mismatch = ((z == 0) ^ is_stuff) & ~ignore
    if bool(mismatch.any()):
        raise AssertionError(
            f"{int(mismatch.sum().item())} pixel(s) violate z(p)==0 <=> s(p) is stuff"
        )


def assert_panoptic_pixel_bijection(sample: DenseSample) -> None:
    """On thing pixels, exactly one instance mask is set per pixel."""
    panoptic = require(sample.panoptic_map, "sample must carry panoptic_map")
    masks = require(sample.instance_masks, "panoptic bijection requires instance_masks")
    if masks.size(0) == 0:
        return

    thing_pixels = torch.as_tensor(panoptic) != 0
    stack = masks.to(torch.int32).sum(dim=0)
    violators = thing_pixels & (stack != 1)
    if bool(violators.any()):
        raise AssertionError(
            f"{int(violators.sum().item())} thing pixel(s) violate"
            " sum_i 1[M_i(p)==1]==1"
        )


def assert_panoptic_fresh_instance_ids(before: DenseSample, after: DenseSample) -> None:
    """Pasted instances get ids strictly greater than ``max_j z_j^b``."""
    before_pan = torch.as_tensor(
        require(before.panoptic_map, "before sample must carry panoptic_map")
    )
    after_pan = torch.as_tensor(
        require(after.panoptic_map, "after sample must carry panoptic_map")
    )

    before_max = int(before_pan.max().item())
    new_ids = set(after_pan.unique().tolist()) - set(before_pan.unique().tolist()) - {0}
    for nid in new_ids:
        if nid <= before_max:
            raise AssertionError(
                f"new instance id {nid} is <= max pre-paste id {before_max}"
            )


def assert_panoptic_stuff_area_threshold(
    before: DenseSample,
    after: DenseSample,
    schema: PanopticSchema,
    tau_stuff: int,
) -> None:
    """A stuff region survives iff its remaining area > ``tau_stuff``."""
    before_s = torch.as_tensor(
        require(before.semantic_map, "before sample must carry semantic_map")
    )
    after_s = torch.as_tensor(
        require(after.semantic_map, "after sample must carry semantic_map")
    )

    for cls, kind in schema.classes.items():
        if kind != "stuff":
            continue
        before_area = int((before_s == cls).sum().item())
        after_area = int((after_s == cls).sum().item())
        if before_area > 0 and after_area == 0 and before_area > tau_stuff:
            raise AssertionError(
                f"stuff class {cls} disappeared but had area"
                f" {before_area} > tau_stuff={tau_stuff}"
            )
        if before_area > 0 and 0 < after_area <= tau_stuff:
            raise AssertionError(
                f"stuff class {cls} survived with area {after_area}"
                f" <= tau_stuff={tau_stuff}"
            )
