"""Dispatch `check_*` predicates against the active modalities of (before, after).

Only invariants with a `(sample)` or `(before, after)` signature are reachable
from the post-hoc viz layer; predicates that need extra context are skipped.
"""

from __future__ import annotations

from segpaste._internal.invariants._report import InvariantReport
from segpaste._internal.invariants.instance import (
    check_instance_identity_preserved,
    check_instance_no_same_class_overlap,
)
from segpaste._internal.invariants.normals import (
    check_normals_camera_frame_convention,
    check_normals_unit_norm_on_valid,
)
from segpaste._internal.invariants.panoptic import check_panoptic_pixel_bijection
from segpaste._internal.invariants.semantic import (
    check_semantic_ignore_preserved,
    check_semantic_single_class_per_pixel,
)
from segpaste.types import DenseSample, Modality


def run_invariants(before: DenseSample, after: DenseSample) -> list[InvariantReport]:
    """Return one :class:`InvariantReport` per applicable check."""
    reports: list[InvariantReport] = []
    active = before.active_modalities() & after.active_modalities()

    if Modality.SEMANTIC in active:
        reports.append(check_semantic_single_class_per_pixel(after))
        reports.append(check_semantic_ignore_preserved(before, after))

    if Modality.INSTANCE in active:
        reports.append(check_instance_no_same_class_overlap(after))
        reports.append(check_instance_identity_preserved(before, after))

    if Modality.PANOPTIC in active and Modality.INSTANCE in active:
        reports.append(check_panoptic_pixel_bijection(after))

    if Modality.NORMALS in active:
        reports.append(check_normals_camera_frame_convention(after))
        reports.append(check_normals_unit_norm_on_valid(after))

    return reports
