"""Dispatch `check_*` predicates against the active modalities of (before, after).

Two-mode behavior:

* ``audit=None`` dispatches the pre-ADR-0014 baseline (7 predicates for a
  sample with all modalities present): the audit-free subset of
  ADR-0001 §Part (ii). This is the bitwise-compatible mode.
* ``audit=BatchAuditPacket`` additionally dispatches 8 more predicates —
  ``bbox_recomputed_from_mask`` and ``fresh_instance_ids`` (whose
  signatures do not technically need audit but are reserved for the
  audited path), plus the 6 audit-using predicates that take the paste
  union, warped source depth, panoptic schema, and fractional thresholds.

Fractional thresholds are converted to absolute pixel counts at this
dispatch site (per ADR-0014 §4) so the existing ``check_*`` signatures —
which take ``tau: int`` — stay untouched.

One invariant is intentionally **not** dispatched here:
``check_depth_metric_intrinsics_rescale`` needs the pre-rescale raw source
depth, which the audit packet does not carry; per ADR-0014 §4, that
invariant is pinned by ``tests/test_invariants_internal.py`` at the
wrapper-level and is carved out of the audit-path dispatch. Audited
coverage is therefore 15 of 16 of the §Part (ii) invariants.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from segpaste._internal.invariants._report import InvariantReport
from segpaste._internal.invariants._require import require
from segpaste._internal.invariants.depth import (
    check_depth_monotonicity,
    check_depth_validity_join,
)
from segpaste._internal.invariants.instance import (
    check_instance_bbox_recomputed_from_mask,
    check_instance_identity_preserved,
    check_instance_no_same_class_overlap,
    check_instance_small_area_dropped,
    check_instance_target_masks_subtract_paste_union,
)
from segpaste._internal.invariants.normals import (
    check_normals_camera_frame_convention,
    check_normals_unit_norm_on_valid,
)
from segpaste._internal.invariants.panoptic import (
    check_panoptic_fresh_instance_ids,
    check_panoptic_pixel_bijection,
    check_panoptic_stuff_area_threshold,
    check_panoptic_thing_stuff_consistent,
)
from segpaste._internal.invariants.semantic import (
    check_semantic_ignore_preserved,
    check_semantic_single_class_per_pixel,
)
from segpaste.types import BatchAuditPacket, DenseSample, Modality


def run_invariants(
    before: DenseSample,
    after: DenseSample,
    audit: BatchAuditPacket | None = None,
) -> list[InvariantReport]:
    """Return one :class:`InvariantReport` per applicable check.

    ``audit`` is optional; when provided, the audit-using invariants
    additionally dispatch. Pass per-sample audit views (``BatchAuditPacket.select(i)``)
    when looping over a batched audit.
    """
    reports: list[InvariantReport] = []
    active = before.active_modalities() & after.active_modalities()

    if Modality.SEMANTIC in active:
        reports.append(check_semantic_single_class_per_pixel(after))
        reports.append(check_semantic_ignore_preserved(before, after))

    if Modality.INSTANCE in active:
        reports.append(check_instance_no_same_class_overlap(after))
        reports.append(check_instance_identity_preserved(before, after))
        if audit is not None:
            reports.append(check_instance_bbox_recomputed_from_mask(after))
            reports.append(
                check_instance_target_masks_subtract_paste_union(
                    before, after, audit.paste_union
                )
            )
            before_masks = require(
                before.instance_masks, "before sample must carry instance_masks"
            )
            before_areas = before_masks.to(torch.bool).flatten(1).sum(dim=1)
            if before_areas.numel() > 0:
                tau = int(
                    audit.thresholds.min_residual_area_frac
                    * float(before_areas.min().item())
                )
                reports.append(check_instance_small_area_dropped(after, tau))

    if Modality.PANOPTIC in active and Modality.INSTANCE in active:
        reports.append(check_panoptic_pixel_bijection(after))

    if Modality.PANOPTIC in active and audit is not None:
        reports.append(check_panoptic_fresh_instance_ids(before, after))

    if (
        Modality.PANOPTIC in active
        and Modality.SEMANTIC in active
        and audit is not None
        and audit.panoptic_schema is not None
    ):
        reports.append(
            check_panoptic_thing_stuff_consistent(after, audit.panoptic_schema)
        )
        if audit.thresholds.tau_stuff_frac is not None:
            sem = require(after.semantic_map, "after sample must carry semantic_map")
            h, w = sem.shape[-2:]
            tau_stuff = int(audit.thresholds.tau_stuff_frac * h * w)
            reports.append(
                check_panoptic_stuff_area_threshold(
                    before, after, audit.panoptic_schema, tau_stuff
                )
            )

    if (
        Modality.DEPTH in active
        and audit is not None
        and audit.warped_source_depth is not None
        and audit.warped_source_depth_valid is not None
    ):
        before_src = replace(
            before,
            depth=audit.warped_source_depth,
            depth_valid=audit.warped_source_depth_valid,
        )
        reports.append(
            check_depth_monotonicity(before_src, before, after, audit.paste_union)
        )
        reports.append(
            check_depth_validity_join(before_src, before, after, audit.paste_union)
        )

    if Modality.NORMALS in active:
        reports.append(check_normals_camera_frame_convention(after))
        reports.append(check_normals_unit_norm_on_valid(after))

    return reports
