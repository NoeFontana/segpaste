"""Unit tests for :class:`BatchAuditPacket` and :class:`AuditThresholds`.

ADR-0014 PR1.
"""

from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError

from segpaste import BatchAuditPacket
from segpaste.types.audit import AuditThresholds


def _make_packet(
    *, with_depth: bool = True, with_intrinsics: bool = True
) -> BatchAuditPacket:
    b, h, w = 2, 4, 4
    intrinsics = torch.tensor([[1.0, 1.0, 2.0, 2.0], [1.5, 1.5, 2.0, 2.0]])
    return BatchAuditPacket(
        paste_union=torch.zeros(b, h, w, dtype=torch.bool),
        warped_source_depth=torch.randn(b, 1, h, w) if with_depth else None,
        warped_source_depth_valid=(
            torch.ones(b, 1, h, w, dtype=torch.bool) if with_depth else None
        ),
        source_intrinsics=intrinsics if with_intrinsics else None,
        panoptic_schema=None,
        thresholds=AuditThresholds(min_residual_area_frac=0.1, tau_stuff_frac=0.05),
    )


def test_named_tuple_positional_access() -> None:
    pkt = _make_packet()
    assert pkt[0] is pkt.paste_union
    assert pkt[5] is pkt.thresholds


def test_select_drops_batch_dim() -> None:
    pkt = _make_packet()
    sample = pkt.select(0)
    assert sample.paste_union.shape == (4, 4)
    assert sample.warped_source_depth is not None
    assert sample.warped_source_depth.shape == (1, 4, 4)
    assert sample.warped_source_depth_valid is not None
    assert sample.warped_source_depth_valid.shape == (1, 4, 4)
    assert sample.source_intrinsics is not None
    assert sample.source_intrinsics.shape == (4,)
    assert sample.thresholds is pkt.thresholds


def test_select_indexes_specific_sample() -> None:
    pkt = _make_packet()
    s0 = pkt.select(0)
    s1 = pkt.select(1)
    assert s0.source_intrinsics is not None
    assert s1.source_intrinsics is not None
    assert float(s0.source_intrinsics[0]) == 1.0
    assert float(s1.source_intrinsics[0]) == 1.5


def test_select_preserves_none_fields() -> None:
    pkt = _make_packet(with_depth=False, with_intrinsics=False)
    sample = pkt.select(0)
    assert sample.warped_source_depth is None
    assert sample.warped_source_depth_valid is None
    assert sample.source_intrinsics is None
    assert sample.paste_union.shape == (4, 4)


def test_audit_thresholds_frozen() -> None:
    th = AuditThresholds(min_residual_area_frac=0.2)
    with pytest.raises(ValidationError):
        th.min_residual_area_frac = 0.5  # type: ignore[misc]


def test_audit_thresholds_extra_forbidden() -> None:
    with pytest.raises(ValidationError):
        AuditThresholds(min_residual_area_frac=0.1, unknown=42)  # type: ignore[call-arg]


def test_audit_thresholds_range_validation() -> None:
    with pytest.raises(ValidationError):
        AuditThresholds(min_residual_area_frac=-0.1)
    with pytest.raises(ValidationError):
        AuditThresholds(min_residual_area_frac=1.5)
    with pytest.raises(ValidationError):
        AuditThresholds(min_residual_area_frac=0.1, tau_stuff_frac=2.0)


def test_audit_thresholds_defaults() -> None:
    th = AuditThresholds(min_residual_area_frac=0.1)
    assert th.tau_stuff_frac is None
    assert th.metric_depth_atol == pytest.approx(1e-3)
