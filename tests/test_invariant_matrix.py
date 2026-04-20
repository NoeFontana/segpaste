"""Invariant matrix — flipping ``xfail=True`` to ``False`` is the gesture
that records a new composite landing on ADR-0001."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pytest
import torch
from _pytest.mark import ParameterSet
from hypothesis import given
from hypothesis import strategies as st

from segpaste.augmentation import CopyPasteAugmentation
from segpaste.config import CopyPasteConfig
from segpaste.types import DenseSample, Modality
from tests.fixtures import fixture_names, load_fixture
from tests.invariants import instance as inv_instance
from tests.strategies import dense_sample_strategy

_MIN_OBJECT_AREA = 1
_NOT_IMPLEMENTED = "composite not yet implemented"

Driver = Callable[[st.DataObject], None]


def _driver_missing_composite(data: st.DataObject) -> None:
    raise NotImplementedError(_NOT_IMPLEMENTED)


@dataclass(frozen=True, slots=True)
class InvariantRow:
    """One ``(modality, invariant-name)`` entry in the ADR-0001 matrix."""

    modality: Modality
    name: str
    driver: Driver = _driver_missing_composite
    xfail: bool = False

    @property
    def test_id(self) -> str:
        return f"{self.modality.value}-{self.name}"


def _augment(sample: DenseSample) -> DenseSample:
    """Run CopyPasteAugmentation round-trip via the DenseSample bridge."""
    cfg = CopyPasteConfig(
        paste_probability=1.0,
        min_paste_objects=1,
        max_paste_objects=2,
        min_object_area=_MIN_OBJECT_AREA,
    )
    aug = CopyPasteAugmentation(cfg)
    target = sample.to_detection_target()
    result = aug.transform(target, [sample.to_detection_target()])
    return DenseSample.from_detection_target(result)


def _driver_instance_bbox(data: st.DataObject) -> None:
    for name in fixture_names({Modality.INSTANCE}):
        inv_instance.assert_instance_bbox_recomputed_from_mask(load_fixture(name))

    sample = data.draw(dense_sample_strategy({Modality.INSTANCE}))
    assert sample.instance_masks is not None
    if sample.instance_masks.size(0) == 0:
        return
    inv_instance.assert_instance_bbox_recomputed_from_mask(_augment(sample))


def _driver_instance_small_area(data: st.DataObject) -> None:
    sample = data.draw(dense_sample_strategy({Modality.INSTANCE}))
    assert sample.instance_masks is not None
    if sample.instance_masks.size(0) == 0:
        return
    inv_instance.assert_instance_small_area_dropped(
        _augment(sample), tau=_MIN_OBJECT_AREA
    )


def _driver_instance_no_same_class_overlap(data: st.DataObject) -> None:  # noqa: ARG001
    for name in fixture_names({Modality.INSTANCE}):
        inv_instance.assert_instance_no_same_class_overlap(load_fixture(name))


def _driver_instance_identity_preserved(data: st.DataObject) -> None:
    sample = data.draw(dense_sample_strategy({Modality.INSTANCE}))
    assert sample.instance_masks is not None
    if sample.instance_masks.size(0) == 0:
        return
    inv_instance.assert_instance_identity_preserved(sample, _augment(sample))


def _driver_instance_target_masks_subtract(data: st.DataObject) -> None:
    sample = data.draw(dense_sample_strategy({Modality.INSTANCE}))
    assert sample.instance_masks is not None
    if sample.instance_masks.size(0) == 0:
        return
    after = _augment(sample)
    assert after.instance_masks is not None
    before_n = sample.instance_masks.size(0)
    # Pasted objects are the extras in `after` beyond the original N.
    if after.instance_masks.size(0) <= before_n:
        return
    paste_union = after.instance_masks[before_n:].to(torch.bool).any(dim=0)
    inv_instance.assert_instance_target_masks_subtract_paste_union(
        sample, after, paste_union
    )


INVARIANT_MATRIX: list[InvariantRow] = [
    InvariantRow(
        Modality.INSTANCE, "identity_preserved", _driver_instance_identity_preserved
    ),
    InvariantRow(
        Modality.INSTANCE,
        "target_masks_subtract_paste_union",
        _driver_instance_target_masks_subtract,
    ),
    InvariantRow(Modality.INSTANCE, "bbox_recomputed_from_mask", _driver_instance_bbox),
    InvariantRow(Modality.INSTANCE, "small_area_dropped", _driver_instance_small_area),
    InvariantRow(
        Modality.INSTANCE,
        "no_same_class_overlap",
        _driver_instance_no_same_class_overlap,
    ),
    InvariantRow(Modality.PANOPTIC, "thing_stuff_consistent", xfail=True),
    InvariantRow(Modality.PANOPTIC, "pixel_bijection", xfail=True),
    InvariantRow(Modality.PANOPTIC, "fresh_instance_ids", xfail=True),
    InvariantRow(Modality.PANOPTIC, "stuff_area_threshold", xfail=True),
    InvariantRow(Modality.SEMANTIC, "single_class_per_pixel", xfail=True),
    InvariantRow(Modality.SEMANTIC, "ignore_preserved", xfail=True),
    InvariantRow(Modality.DEPTH, "monotonicity", xfail=True),
    InvariantRow(Modality.DEPTH, "validity_join", xfail=True),
    InvariantRow(Modality.DEPTH, "metric_intrinsics_rescale", xfail=True),
    InvariantRow(Modality.NORMALS, "unit_norm_on_valid", xfail=True),
    InvariantRow(Modality.NORMALS, "camera_frame_convention", xfail=True),
]


def _params() -> list[ParameterSet]:
    out: list[ParameterSet] = []
    for row in INVARIANT_MATRIX:
        marks: list[pytest.MarkDecorator] = []
        if row.xfail:
            marks.append(
                pytest.mark.xfail(
                    reason=_NOT_IMPLEMENTED,
                    strict=True,
                    raises=NotImplementedError,
                )
            )
        out.append(pytest.param(row, id=row.test_id, marks=marks))
    return out


@pytest.mark.invariant
@pytest.mark.parametrize("row", _params())
@given(data=st.data())
def test_invariant_matrix(row: InvariantRow, data: st.DataObject) -> None:
    row.driver(data)
