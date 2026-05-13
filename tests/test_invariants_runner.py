"""Dispatch-count tests for :func:`run_invariants` (ADR-0014 PR4).

Verifies that the runner dispatches the right subset of ADR-0001 §Part (ii)
invariants for each (modalities, audit-present?) combination. The audit-using
predicates already exist under ``segpaste._internal.invariants/``; PR4 only
wires their dispatch — no predicate authorship.
"""

from __future__ import annotations

import torch

from segpaste._internal.viz.invariant_runner import run_invariants
from segpaste.types import BatchAuditPacket, DenseSample
from segpaste.types.audit import AuditThresholds
from segpaste.types.dense_sample import PanopticSchemaSpec
from tests.fixtures.synthetic.builders import (
    build_depth_two_planes,
    build_panoptic_stuff_and_things,
    build_two_overlapping_things,
)


def _per_sample_audit(
    sample: DenseSample,
    *,
    panoptic_schema: PanopticSchemaSpec | None = None,
    tau_stuff_frac: float | None = None,
    with_warped_depth: bool = False,
) -> BatchAuditPacket:
    h, w = sample.image.shape[-2:]
    paste_union = torch.zeros((h, w), dtype=torch.bool)
    warped_depth = sample.depth if with_warped_depth else None
    warped_valid = sample.depth_valid if with_warped_depth else None
    return BatchAuditPacket(
        paste_union=paste_union,
        warped_source_depth=warped_depth,
        warped_source_depth_valid=warped_valid,
        source_intrinsics=None,
        panoptic_schema=panoptic_schema,
        thresholds=AuditThresholds(
            min_residual_area_frac=0.1,
            tau_stuff_frac=tau_stuff_frac,
        ),
    )


class TestNoAuditDispatch:
    """``audit=None`` dispatches the pre-ADR-0014 baseline (bitwise compat)."""

    def test_instance_only_dispatches_two(self) -> None:
        sample = build_two_overlapping_things()
        reports = run_invariants(sample, sample)
        assert sorted(r.name for r in reports) == [
            "instance.identity_preserved",
            "instance.no_same_class_overlap",
        ]

    def test_panoptic_sample_dispatches_five(self) -> None:
        """SEMANTIC + INSTANCE + PANOPTIC active → 5 reports (baseline)."""
        sample = build_panoptic_stuff_and_things()
        reports = run_invariants(sample, sample)
        names = sorted(r.name for r in reports)
        assert names == [
            "instance.identity_preserved",
            "instance.no_same_class_overlap",
            "panoptic.pixel_bijection",
            "semantic.ignore_preserved",
            "semantic.single_class_per_pixel",
        ]

    def test_fresh_instance_ids_not_dispatched_without_audit(self) -> None:
        """``fresh_instance_ids`` is reserved for the audited path."""
        sample = build_panoptic_stuff_and_things()
        reports = run_invariants(sample, sample)
        assert not any(r.name == "panoptic.fresh_instance_ids" for r in reports)

    def test_bbox_recomputed_not_dispatched_without_audit(self) -> None:
        """``bbox_recomputed_from_mask`` is reserved for the audited path."""
        sample = build_two_overlapping_things()
        reports = run_invariants(sample, sample)
        assert not any(r.name == "instance.bbox_recomputed_from_mask" for r in reports)


class TestAuditDispatch:
    """With ``audit``, dispatch adds the context-dependent predicates."""

    def test_instance_audit_adds_two(self) -> None:
        sample = build_two_overlapping_things()
        audit = _per_sample_audit(sample)
        reports = run_invariants(sample, sample, audit=audit)
        names = sorted(r.name for r in reports)
        assert names == [
            "instance.bbox_recomputed_from_mask",
            "instance.identity_preserved",
            "instance.no_same_class_overlap",
            "instance.small_area_dropped",
            "instance.target_masks_subtract_paste_union",
        ]

    def test_panoptic_with_schema_adds_thing_stuff_and_stuff_area(self) -> None:
        sample = build_panoptic_stuff_and_things()
        schema = PanopticSchemaSpec(
            classes={0: "stuff", 1: "thing", 2: "thing"},
        )
        audit = _per_sample_audit(sample, panoptic_schema=schema, tau_stuff_frac=0.05)
        reports = run_invariants(sample, sample, audit=audit)
        assert {r.name for r in reports} == {
            "semantic.single_class_per_pixel",
            "semantic.ignore_preserved",
            "instance.no_same_class_overlap",
            "instance.identity_preserved",
            "instance.bbox_recomputed_from_mask",
            "instance.target_masks_subtract_paste_union",
            "instance.small_area_dropped",
            "panoptic.pixel_bijection",
            "panoptic.fresh_instance_ids",
            "panoptic.thing_stuff_consistent",
            "panoptic.stuff_area_threshold",
        }

    def test_panoptic_without_schema_skips_audit_predicates(self) -> None:
        """Audit present but schema=None → audit-using panoptic predicates skip."""
        sample = build_panoptic_stuff_and_things()
        audit = _per_sample_audit(sample, panoptic_schema=None)
        reports = run_invariants(sample, sample, audit=audit)
        names = {r.name for r in reports}
        assert "panoptic.thing_stuff_consistent" not in names
        assert "panoptic.stuff_area_threshold" not in names

    def test_panoptic_with_schema_no_tau_skips_stuff_area_only(self) -> None:
        """When schema is set but tau_stuff_frac is None, stuff_area still skipped."""
        sample = build_panoptic_stuff_and_things()
        schema = PanopticSchemaSpec(classes={0: "stuff", 1: "thing", 2: "thing"})
        audit = _per_sample_audit(sample, panoptic_schema=schema, tau_stuff_frac=None)
        reports = run_invariants(sample, sample, audit=audit)
        names = {r.name for r in reports}
        assert "panoptic.thing_stuff_consistent" in names
        assert "panoptic.stuff_area_threshold" not in names

    def test_depth_audit_adds_two(self) -> None:
        sample = build_depth_two_planes()
        audit = _per_sample_audit(sample, with_warped_depth=True)
        reports = run_invariants(sample, sample, audit=audit)
        names = {r.name for r in reports}
        assert "depth.monotonicity" in names
        assert "depth.validity_join" in names

    def test_depth_audit_without_warped_depth_skips(self) -> None:
        """Audit present but no warped source depth → depth predicates skip."""
        sample = build_depth_two_planes()
        audit = _per_sample_audit(sample, with_warped_depth=False)
        reports = run_invariants(sample, sample, audit=audit)
        names = {r.name for r in reports}
        assert "depth.monotonicity" not in names
        assert "depth.validity_join" not in names

    def test_metric_intrinsics_rescale_is_never_dispatched(self) -> None:
        """ADR-0014 §4 carve-out: not dispatched from the runner."""
        sample = build_depth_two_planes()
        audit = _per_sample_audit(sample, with_warped_depth=True)
        reports = run_invariants(sample, sample, audit=audit)
        names = {r.name for r in reports}
        assert "depth.metric_intrinsics_rescale" not in names


class TestBackwardCompatibility:
    """``audit=None`` preserves the pre-ADR-0014 behavior for the base cases."""

    def test_default_kwarg_is_none(self) -> None:
        sample = build_two_overlapping_things()
        positional = run_invariants(sample, sample)
        kwargged = run_invariants(sample, sample, audit=None)
        assert [r.name for r in positional] == [r.name for r in kwargged]
