"""Semantic-modality invariants per ADR-0001 §(ii)."""

import torch

from segpaste.types import DenseSample
from tests.invariants import require

_IGNORE_LABEL = 255


def assert_semantic_single_class_per_pixel(sample: DenseSample) -> None:
    """Semantic map is a single ``[H, W]`` int tensor (no multi-channel)."""
    tensor = torch.as_tensor(
        require(sample.semantic_map, "sample must carry semantic_map")
    )
    if tensor.ndim != 2:
        raise AssertionError(
            f"semantic_map must be [H, W]; got shape {tuple(tensor.shape)}"
        )
    if tensor.dtype.is_floating_point or tensor.dtype == torch.bool:
        raise AssertionError(
            f"semantic_map dtype {tensor.dtype} is not integer;"
            " multi-channel semantics are not allowed"
        )


def assert_semantic_ignore_preserved(before: DenseSample, after: DenseSample) -> None:
    """Every pixel that was ``255`` in ``before`` is still ``255`` in ``after``."""
    before_s = torch.as_tensor(
        require(before.semantic_map, "before sample must carry semantic_map")
    )
    after_s = torch.as_tensor(
        require(after.semantic_map, "after sample must carry semantic_map")
    )
    if before_s.shape != after_s.shape:
        raise AssertionError(
            f"semantic_map shape changed: {tuple(before_s.shape)}"
            f" -> {tuple(after_s.shape)}"
        )
    violated = (before_s == _IGNORE_LABEL) & (after_s != _IGNORE_LABEL)
    if bool(violated.any()):
        raise AssertionError(
            f"{int(violated.sum().item())} ignore pixel(s) overwritten by paste"
        )
