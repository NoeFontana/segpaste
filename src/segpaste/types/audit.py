"""Optional sidecar return type from :meth:`BatchCopyPaste.forward_with_audit`.

Per ADR-0014: carries the post-z-test paste union, warped source depth fields,
gathered source intrinsics, panoptic schema, and area thresholds needed by the
audit-using subset of ADR-0001 §Part (ii) invariants. Consumed by
``segpaste._internal.viz.invariant_runner.run_invariants(before, after, audit=...)``
in the visualizer's offline path; the training hot path returns only the sample
via :meth:`BatchCopyPaste.forward` and never constructs this object.

NamedTuple (not dataclass) so construction traces as a tuple op under
``torch.compile`` — relevant to keep the audit assembly graph-clean if a future
caller ever inlines it. Today only ``forward_with_audit`` constructs it, and
that method is explicitly excluded from compile-clean tracing per ADR-0014 §7.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple, TypeVar

import torch
from pydantic import BaseModel, ConfigDict, Field

from segpaste.types.dense_sample import PanopticSchemaSpec

_T = TypeVar("_T")


def _map_optional(x: _T | None, fn: Callable[[_T], _T]) -> _T | None:
    return None if x is None else fn(x)


class AuditThresholds(BaseModel):
    """Fractional thresholds consumed by audited invariant dispatch.

    ``run_invariants`` converts these to absolute pixel counts at dispatch
    time so the existing predicates' ``tau: int`` signatures stay untouched.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_residual_area_frac: float = Field(ge=0.0, le=1.0)
    tau_stuff_frac: float | None = Field(default=None, ge=0.0, le=1.0)
    metric_depth_atol: float = Field(default=1e-3, ge=0.0)


class BatchAuditPacket(NamedTuple):
    """Sidecar bundle returned by :meth:`BatchCopyPaste.forward_with_audit`.

    ``paste_union`` is always populated (OR-reduce over per-tile effective
    paste masks). Every other tensor field is ``None`` when its modality is
    absent. ``panoptic_schema`` is ``None`` when the composite is not in
    panoptic mode. ``thresholds`` is always populated.

    Shapes are leading-batch (``B``) at construction. :meth:`select` drops
    the leading batch dim for per-sample dispatch into ``run_invariants``.
    """

    paste_union: torch.Tensor
    warped_source_depth: torch.Tensor | None
    warped_source_depth_valid: torch.Tensor | None
    source_intrinsics: torch.Tensor | None
    panoptic_schema: PanopticSchemaSpec | None
    thresholds: AuditThresholds

    def to(self, device: torch.device) -> BatchAuditPacket:
        """Return a copy with every tensor field moved to ``device``."""

        def move(t: torch.Tensor) -> torch.Tensor:
            return t.to(device)

        return BatchAuditPacket(
            paste_union=move(self.paste_union),
            warped_source_depth=_map_optional(self.warped_source_depth, move),
            warped_source_depth_valid=_map_optional(
                self.warped_source_depth_valid, move
            ),
            source_intrinsics=_map_optional(self.source_intrinsics, move),
            panoptic_schema=self.panoptic_schema,
            thresholds=self.thresholds,
        )

    def select(self, i: int) -> BatchAuditPacket:
        """Return a per-sample view by indexing every batch-leading tensor.

        The leading batch dim is dropped; ``[B, H, W] -> [H, W]`` and
        ``[B, 1, H, W] -> [1, H, W]``. ``panoptic_schema`` and ``thresholds``
        are static and passed through by reference.
        """

        def pick(t: torch.Tensor) -> torch.Tensor:
            return t[i]

        return BatchAuditPacket(
            paste_union=pick(self.paste_union),
            warped_source_depth=_map_optional(self.warped_source_depth, pick),
            warped_source_depth_valid=_map_optional(
                self.warped_source_depth_valid, pick
            ),
            source_intrinsics=_map_optional(self.source_intrinsics, pick),
            panoptic_schema=self.panoptic_schema,
            thresholds=self.thresholds,
        )
