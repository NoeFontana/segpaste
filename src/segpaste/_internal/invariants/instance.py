"""Instance-modality invariants per ADR-0001 §(ii)."""

from __future__ import annotations

import torch
from torchvision.ops import masks_to_boxes

from segpaste._internal.invariants._report import InvariantReport, raise_if_violated
from segpaste._internal.invariants._require import require
from segpaste.processing import compute_mask_area
from segpaste.types import DenseSample


def check_instance_identity_preserved(
    before: DenseSample, after: DenseSample
) -> InvariantReport:
    """Every surviving instance keeps its label.

    Identity preservation here is label-level: any ``after`` instance whose
    mask is contained by at least one ``before`` instance mask is treated as
    a surviving original and must share a label with some such container.
    After-instances not contained by any before-mask are new pastes and are
    ignored. When multiple before-instances contain the after-mask (e.g.
    overlapping/identical before-masks), matching any one of their labels
    suffices — we cannot disambiguate parentage from masks alone.
    """
    name = "instance.identity_preserved"
    before_masks = require(
        before.instance_masks, "before sample must carry instance_masks"
    ).to(torch.bool)
    after_masks = require(
        after.instance_masks, "after sample must carry instance_masks"
    ).to(torch.bool)

    if after_masks.numel() == 0:
        return InvariantReport(name=name, ok=True)

    for j in range(after_masks.size(0)):
        after_mask = after_masks[j]
        if not after_mask.any():
            continue
        after_label = int(after.labels[j].item())
        containing: list[int] = []
        for i in range(before_masks.size(0)):
            before_mask = before_masks[i]
            if before_mask.any() and bool(torch.all(after_mask <= before_mask)):
                containing.append(i)
        if not containing:
            continue
        if not any(int(before.labels[i].item()) == after_label for i in containing):
            return InvariantReport(
                name=name,
                ok=False,
                message=(
                    f"after-instance {j} (label {after_label}) is contained by"
                    f" before-instances {containing} but none share its label"
                ),
                details={"after_index": j, "after_label": after_label},
            )

    return InvariantReport(name=name, ok=True)


def assert_instance_identity_preserved(before: DenseSample, after: DenseSample) -> None:
    raise_if_violated(check_instance_identity_preserved(before, after))


def check_instance_target_masks_subtract_paste_union(
    before: DenseSample, after: DenseSample, paste_union: torch.Tensor
) -> InvariantReport:
    """Surviving target masks equal ``M_i \\ U`` where ``U`` is the paste union.

    Operationally: every non-empty after-mask is either entirely outside U
    (a surviving original, whose pixels in U were subtracted) or entirely
    inside U (a pasted instance, whose mask contributes to U by construction).
    A mask straddling U would mean the subtraction was incomplete.
    """
    name = "instance.target_masks_subtract_paste_union"
    require(before.instance_masks, "before sample must carry instance_masks")
    after_masks = require(
        after.instance_masks, "after sample must carry instance_masks"
    ).to(torch.bool)
    u = paste_union.to(torch.bool)

    nonempty = after_masks.flatten(1).any(dim=1)
    has_inside = (after_masks & u).flatten(1).any(dim=1)
    has_outside = (after_masks & ~u).flatten(1).any(dim=1)
    straddles = nonempty & has_inside & has_outside
    if bool(straddles.any()):
        idx = int(straddles.nonzero(as_tuple=True)[0][0].item())
        return InvariantReport(
            name=name,
            ok=False,
            message=(
                f"after-instance {idx} straddles the paste union U"
                " (expected either fully inside = paste, or fully outside = survivor);"
                " subtraction by U must be complete on survivors"
            ),
            details={"straddling_index": idx},
        )
    return InvariantReport(name=name, ok=True)


def assert_instance_target_masks_subtract_paste_union(
    before: DenseSample, after: DenseSample, paste_union: torch.Tensor
) -> None:
    raise_if_violated(
        check_instance_target_masks_subtract_paste_union(before, after, paste_union)
    )


def check_instance_bbox_recomputed_from_mask(sample: DenseSample) -> InvariantReport:
    """Each box equals ``bbox(mask)`` for its instance."""
    name = "instance.bbox_recomputed_from_mask"
    masks = require(sample.instance_masks, "sample must carry instance_masks")
    if masks.size(0) == 0:
        return InvariantReport(name=name, ok=True)

    nonempty = masks.to(torch.bool).flatten(1).any(dim=1)
    if not bool(nonempty.any()):
        return InvariantReport(name=name, ok=True)

    computed = masks_to_boxes(masks[nonempty].to(torch.uint8))
    actual = torch.as_tensor(sample.boxes)[nonempty].to(computed.dtype)
    if not torch.allclose(computed, actual, atol=1.0):
        return InvariantReport(
            name=name,
            ok=False,
            message=(
                "boxes disagree with masks_to_boxes(masks):\n"
                f"  computed: {computed}\n  actual: {actual}"
            ),
        )
    return InvariantReport(name=name, ok=True)


def assert_instance_bbox_recomputed_from_mask(sample: DenseSample) -> None:
    raise_if_violated(check_instance_bbox_recomputed_from_mask(sample))


def check_instance_small_area_dropped(sample: DenseSample, tau: int) -> InvariantReport:
    """No surviving instance has mask area < ``tau``."""
    name = "instance.small_area_dropped"
    masks = require(sample.instance_masks, "sample must carry instance_masks")
    if masks.size(0) == 0:
        return InvariantReport(name=name, ok=True)

    areas = compute_mask_area(masks.to(torch.bool))
    too_small = areas < tau
    if bool(too_small.any()):
        violator_count = int(too_small.sum().item())
        return InvariantReport(
            name=name,
            ok=False,
            message=(
                f"{violator_count} instance(s) have area < {tau}:"
                f" areas={areas.tolist()}"
            ),
            details={"violator_count": violator_count, "tau": tau},
        )
    return InvariantReport(name=name, ok=True)


def assert_instance_small_area_dropped(sample: DenseSample, tau: int) -> None:
    raise_if_violated(check_instance_small_area_dropped(sample, tau))


def check_instance_no_same_class_overlap(sample: DenseSample) -> InvariantReport:
    """No pixel is assigned to two instances of the same class."""
    name = "instance.no_same_class_overlap"
    masks_opt = require(sample.instance_masks, "sample must carry instance_masks")
    if masks_opt.size(0) < 2:
        return InvariantReport(name=name, ok=True)

    masks = masks_opt.to(torch.bool)
    labels = sample.labels
    for cls in torch.unique(labels).tolist():
        cls_masks = masks[labels == cls]
        if cls_masks.size(0) < 2:
            continue
        if bool((cls_masks.to(torch.int32).sum(dim=0) > 1).any()):
            return InvariantReport(
                name=name,
                ok=False,
                message=f"class {cls} has pixels assigned to >1 instance",
                details={"class": int(cls)},
            )
    return InvariantReport(name=name, ok=True)


def assert_instance_no_same_class_overlap(sample: DenseSample) -> None:
    raise_if_violated(check_instance_no_same_class_overlap(sample))
