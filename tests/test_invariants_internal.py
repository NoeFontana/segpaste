"""Smoke tests for `segpaste._internal.invariants` ``check_*`` predicates.

Each test draws (or constructs) a structurally-valid input that the
invariant should accept, calls the corresponding ``check_*``, and asserts
``report.ok is True``. The point is to exercise the call path on every
modality after the move from ``tests/invariants/``; the violation logic
itself is covered indirectly via :mod:`tests.test_huggingface_roundtrip`
and the future visualizer (P5+).
"""

from __future__ import annotations

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from segpaste._internal.invariants.depth import (
    check_depth_metric_intrinsics_rescale,
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
from segpaste.types import CameraIntrinsics, Modality, PanopticSchema
from tests.shared import FakePanopticSchema, make_disjoint_panoptic_sample
from tests.strategies.dense_sample import dense_sample_strategy


def _fuzz_schema() -> PanopticSchema:
    """Schema matching `tests.strategies.dense_sample` panoptic conventions."""
    return FakePanopticSchema(
        classes={i: ("stuff" if i == 0 else "thing") for i in range(8)},
        ignore_index=255,
        max_instances_per_image=256,
    )


# ---- instance ----------------------------------------------------------


@settings(deadline=None, max_examples=50)
@given(data=st.data())
def test_check_instance_identity_preserved_passes(data: st.DataObject) -> None:
    sample = data.draw(dense_sample_strategy({Modality.INSTANCE}))
    report = check_instance_identity_preserved(sample, sample)
    assert report.ok, report.message


@settings(deadline=None, max_examples=50)
@given(data=st.data())
def test_check_instance_target_masks_subtract_paste_union_passes(
    data: st.DataObject,
) -> None:
    sample = data.draw(dense_sample_strategy({Modality.INSTANCE}))
    h, w = sample.image.shape[-2:]
    paste_union = torch.zeros((h, w), dtype=torch.bool)
    report = check_instance_target_masks_subtract_paste_union(
        sample, sample, paste_union
    )
    assert report.ok, report.message


@settings(deadline=None, max_examples=50)
@given(data=st.data())
def test_check_instance_bbox_recomputed_from_mask_passes(data: st.DataObject) -> None:
    sample = data.draw(dense_sample_strategy({Modality.INSTANCE}))
    report = check_instance_bbox_recomputed_from_mask(sample)
    assert report.ok, report.message


@settings(deadline=None, max_examples=50)
@given(data=st.data())
def test_check_instance_small_area_dropped_passes(data: st.DataObject) -> None:
    sample = data.draw(dense_sample_strategy({Modality.INSTANCE}))
    report = check_instance_small_area_dropped(sample, tau=1)
    assert report.ok, report.message


def test_check_instance_no_same_class_overlap_passes() -> None:
    sample = make_disjoint_panoptic_sample()
    report = check_instance_no_same_class_overlap(sample)
    assert report.ok, report.message


# ---- panoptic ----------------------------------------------------------


@settings(deadline=None, max_examples=50)
@given(data=st.data())
def test_check_panoptic_thing_stuff_consistent_passes(data: st.DataObject) -> None:
    sample = data.draw(dense_sample_strategy({Modality.SEMANTIC, Modality.PANOPTIC}))
    report = check_panoptic_thing_stuff_consistent(sample, _fuzz_schema())
    assert report.ok, report.message


def test_check_panoptic_pixel_bijection_passes() -> None:
    sample = make_disjoint_panoptic_sample()
    report = check_panoptic_pixel_bijection(sample)
    assert report.ok, report.message


@settings(deadline=None, max_examples=50)
@given(data=st.data())
def test_check_panoptic_fresh_instance_ids_passes(data: st.DataObject) -> None:
    sample = data.draw(dense_sample_strategy({Modality.PANOPTIC}))
    report = check_panoptic_fresh_instance_ids(sample, sample)
    assert report.ok, report.message


@settings(deadline=None, max_examples=50)
@given(data=st.data())
def test_check_panoptic_stuff_area_threshold_passes(data: st.DataObject) -> None:
    sample = data.draw(dense_sample_strategy({Modality.SEMANTIC, Modality.PANOPTIC}))
    report = check_panoptic_stuff_area_threshold(
        sample, sample, _fuzz_schema(), tau_stuff=0
    )
    assert report.ok, report.message


# ---- semantic ----------------------------------------------------------


@settings(deadline=None, max_examples=50)
@given(data=st.data())
def test_check_semantic_single_class_per_pixel_passes(data: st.DataObject) -> None:
    sample = data.draw(dense_sample_strategy({Modality.SEMANTIC}))
    report = check_semantic_single_class_per_pixel(sample)
    assert report.ok, report.message


@settings(deadline=None, max_examples=50)
@given(data=st.data())
def test_check_semantic_ignore_preserved_passes(data: st.DataObject) -> None:
    sample = data.draw(dense_sample_strategy({Modality.SEMANTIC}))
    report = check_semantic_ignore_preserved(sample, sample)
    assert report.ok, report.message


# ---- depth -------------------------------------------------------------


@settings(deadline=None, max_examples=50)
@given(data=st.data())
def test_check_depth_monotonicity_passes(data: st.DataObject) -> None:
    sample = data.draw(dense_sample_strategy({Modality.DEPTH}))
    h, w = sample.image.shape[-2:]
    paste_mask = torch.zeros((1, h, w), dtype=torch.bool)
    report = check_depth_monotonicity(sample, sample, sample, paste_mask)
    assert report.ok, report.message


@settings(deadline=None, max_examples=50)
@given(data=st.data())
def test_check_depth_validity_join_passes(data: st.DataObject) -> None:
    sample = data.draw(dense_sample_strategy({Modality.DEPTH}))
    h, w = sample.image.shape[-2:]
    paste_mask = torch.zeros((1, h, w), dtype=torch.bool)
    report = check_depth_validity_join(sample, sample, sample, paste_mask)
    assert report.ok, report.message


@settings(deadline=None, max_examples=50)
@given(
    fx_s=st.floats(min_value=100.0, max_value=2000.0, allow_nan=False),
    fy_s=st.floats(min_value=100.0, max_value=2000.0, allow_nan=False),
    fx_t=st.floats(min_value=100.0, max_value=2000.0, allow_nan=False),
    fy_t=st.floats(min_value=100.0, max_value=2000.0, allow_nan=False),
)
def test_check_depth_metric_intrinsics_rescale_passes(
    fx_s: float, fy_s: float, fx_t: float, fy_t: float
) -> None:
    src = CameraIntrinsics(fx=fx_s, fy=fy_s, cx=12.0, cy=12.0)
    tgt = CameraIntrinsics(fx=fx_t, fy=fy_t, cx=12.0, cy=12.0)
    raw = torch.rand(1, 24, 24)
    ratio = ((fx_t * fy_t) ** 0.5) / ((fx_s * fy_s) ** 0.5)
    rescaled = raw * ratio
    report = check_depth_metric_intrinsics_rescale(raw, src, tgt, rescaled)
    assert report.ok, report.message


# ---- normals -----------------------------------------------------------


@settings(deadline=None, max_examples=50)
@given(data=st.data())
def test_check_normals_unit_norm_on_valid_passes(data: st.DataObject) -> None:
    sample = data.draw(dense_sample_strategy({Modality.NORMALS}))
    report = check_normals_unit_norm_on_valid(sample)
    assert report.ok, report.message


@settings(deadline=None, max_examples=50)
@given(data=st.data())
def test_check_normals_camera_frame_convention_passes(data: st.DataObject) -> None:
    sample = data.draw(dense_sample_strategy({Modality.NORMALS}))
    report = check_normals_camera_frame_convention(sample)
    assert report.ok, report.message
