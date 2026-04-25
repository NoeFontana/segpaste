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
    paste_mask: torch.Tensor,
) -> None:
    """Piecewise validity join (ADR-0001 §(ii), ADR-0007 §5).

    Inside the translated paste mask, ``V_out = V_src ∧ V_tgt``; outside,
    ``V_out = V_tgt``. ``paste_mask`` is the footprint the source was
    pasted into (pre-z-test) — the composite is still target-dominant
    outside that footprint.
    """
    v_src = _depth_valid(before_src, "before_src")
    v_tgt = _depth_valid(before_tgt, "before_tgt")
    v_out = _depth_valid(after, "after")

    p = paste_mask.to(torch.bool)
    if p.dim() == 2:
        p = p.unsqueeze(0)
    expected = torch.where(p, v_src & v_tgt, v_tgt)
    if not torch.equal(v_out, expected):
        raise AssertionError(
            "depth_valid violates piecewise join: expected V_src & V_tgt"
            " inside paste_mask and V_tgt outside"
        )


def assert_depth_metric_intrinsics_rescale(
    src_raw_depth: torch.Tensor,
    src_intrinsics: CameraIntrinsics,
    tgt_intrinsics: CameraIntrinsics,
    rescaled_depth: torch.Tensor,
    *,
    atol: float = 1e-5,
) -> None:
    """``d_src <- d_src * sqrt(fx_t*fy_t) / sqrt(fx_s*fy_s)`` (ADR-0007 §4).

    The geometric mean of ``fx`` and ``fy`` handles non-square pixels
    symmetrically; for isotropic pixels it reduces to the ``f_t/f_s``
    ratio from ADR-0001 §(ii).
    """
    num = (tgt_intrinsics.fx * tgt_intrinsics.fy) ** 0.5
    den = (src_intrinsics.fx * src_intrinsics.fy) ** 0.5
    ratio = num / den
    expected = src_raw_depth * ratio
    if not torch.allclose(rescaled_depth, expected, atol=atol):
        raise AssertionError(
            f"metric-depth rescale wrong: expected d * {ratio}, got"
            f" max-abs-err {(rescaled_depth - expected).abs().max().item()}"
        )
