"""Semantic-modality invariants per ADR-0001 §(ii)."""

from __future__ import annotations

import torch

from segpaste._internal.invariants._report import InvariantReport, raise_if_violated
from segpaste._internal.invariants._require import require
from segpaste.types import DenseSample

_IGNORE_LABEL = 255


def check_semantic_single_class_per_pixel(sample: DenseSample) -> InvariantReport:
    """Semantic map is a single ``[H, W]`` int tensor (no multi-channel)."""
    name = "semantic.single_class_per_pixel"
    tensor = torch.as_tensor(
        require(sample.semantic_map, "sample must carry semantic_map")
    )
    if tensor.ndim != 2:
        return InvariantReport(
            name=name,
            ok=False,
            message=f"semantic_map must be [H, W]; got shape {tuple(tensor.shape)}",
            details={"shape": str(tuple(tensor.shape))},
        )
    if tensor.dtype.is_floating_point or tensor.dtype == torch.bool:
        return InvariantReport(
            name=name,
            ok=False,
            message=(
                f"semantic_map dtype {tensor.dtype} is not integer;"
                " multi-channel semantics are not allowed"
            ),
            details={"dtype": str(tensor.dtype)},
        )
    return InvariantReport(name=name, ok=True)


def assert_semantic_single_class_per_pixel(sample: DenseSample) -> None:
    raise_if_violated(check_semantic_single_class_per_pixel(sample))


def check_semantic_ignore_preserved(
    before: DenseSample, after: DenseSample
) -> InvariantReport:
    """Every pixel that was ``255`` in ``before`` is still ``255`` in ``after``."""
    name = "semantic.ignore_preserved"
    before_s = torch.as_tensor(
        require(before.semantic_map, "before sample must carry semantic_map")
    )
    after_s = torch.as_tensor(
        require(after.semantic_map, "after sample must carry semantic_map")
    )
    if before_s.shape != after_s.shape:
        return InvariantReport(
            name=name,
            ok=False,
            message=(
                f"semantic_map shape changed: {tuple(before_s.shape)}"
                f" -> {tuple(after_s.shape)}"
            ),
            details={
                "shape_before": str(tuple(before_s.shape)),
                "shape_after": str(tuple(after_s.shape)),
            },
        )
    violated = (before_s == _IGNORE_LABEL) & (after_s != _IGNORE_LABEL)
    if bool(violated.any()):
        violator_count = int(violated.sum().item())
        return InvariantReport(
            name=name,
            ok=False,
            message=f"{violator_count} ignore pixel(s) overwritten by paste",
            details={"violator_count": violator_count},
        )
    return InvariantReport(name=name, ok=True)


def assert_semantic_ignore_preserved(before: DenseSample, after: DenseSample) -> None:
    raise_if_violated(check_semantic_ignore_preserved(before, after))
