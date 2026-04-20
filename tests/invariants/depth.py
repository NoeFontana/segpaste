"""Depth-modality invariants per ADR-0001 §(ii)."""

import torch

from segpaste.types import CameraIntrinsics, DenseSample
from tests.invariants import require


def _depth(sample: DenseSample, side: str) -> torch.Tensor:
    return require(sample.depth, f"{side} sample must carry depth")


def _depth_valid(sample: DenseSample, side: str) -> torch.Tensor:
    return require(sample.depth_valid, f"{side} sample must carry depth_valid")


def assert_depth_monotonicity(
    before_src: DenseSample,
    before_tgt: DenseSample,
    after: DenseSample,
    effective_paste_mask: torch.Tensor,
) -> None:
    """``d_out = min(d_src, d_tgt)`` inside the effective paste mask."""
    d_src = _depth(before_src, "before_src")
    d_tgt = _depth(before_tgt, "before_tgt")
    d_out = _depth(after, "after")

    mask = effective_paste_mask.to(torch.bool)
    if not bool(mask.any()):
        return
    mismatch = (d_out != torch.minimum(d_src, d_tgt)) & mask
    if bool(mismatch.any()):
        raise AssertionError(
            f"{int(mismatch.sum().item())} pixel(s) violate"
            " d_out = min(d_src, d_tgt) inside M_eff"
        )


def assert_depth_validity_join(
    before_src: DenseSample,
    before_tgt: DenseSample,
    after: DenseSample,
) -> None:
    """``V_out = V_src & V_tgt`` pixelwise."""
    v_src = _depth_valid(before_src, "before_src")
    v_tgt = _depth_valid(before_tgt, "before_tgt")
    v_out = _depth_valid(after, "after")

    if not torch.equal(v_out, v_src & v_tgt):
        raise AssertionError("depth_valid != V_src & V_tgt")


def assert_depth_metric_intrinsics_rescale(
    src_raw_depth: torch.Tensor,
    src_intrinsics: CameraIntrinsics,
    tgt_intrinsics: CameraIntrinsics,
    rescaled_depth: torch.Tensor,
    *,
    atol: float = 1e-5,
) -> None:
    """``d_src <- d_src * f_tgt / f_src`` when metric depth is enabled.

    Uses ``fx`` as the reference focal length; composites that need a
    different axis should implement their own check.
    """
    ratio = tgt_intrinsics.fx / src_intrinsics.fx
    expected = src_raw_depth * ratio
    if not torch.allclose(rescaled_depth, expected, atol=atol):
        raise AssertionError(
            f"metric-depth rescale wrong: expected d * {ratio}, got"
            f" max-abs-err {(rescaled_depth - expected).abs().max().item()}"
        )
