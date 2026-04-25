"""Normals-modality invariants per ADR-0001 §(ii)."""

from __future__ import annotations

import torch

from segpaste._internal.invariants._report import InvariantReport, raise_if_violated
from segpaste._internal.invariants._require import require
from segpaste.types import DenseSample


def check_normals_unit_norm_on_valid(
    sample: DenseSample, *, atol: float = 1e-5
) -> InvariantReport:
    """``||n(p)||_2 == 1`` on every valid pixel within ``atol``."""
    name = "normals.unit_norm_on_valid"
    normals = require(sample.normals, "sample must carry normals")
    norms = normals.norm(dim=0)
    if sample.depth_valid is not None:
        valid = sample.depth_valid.squeeze(0).to(torch.bool)
    else:
        valid = torch.ones_like(norms, dtype=torch.bool)
    if not bool(valid.any()):
        return InvariantReport(name=name, ok=True)
    err = (norms[valid] - 1.0).abs()
    if bool((err > atol).any()):
        max_error = float(err.max().item())
        return InvariantReport(
            name=name,
            ok=False,
            message=f"normals violate unit norm (max error {max_error:.3e})",
            details={"max_error": max_error},
        )
    return InvariantReport(name=name, ok=True)


def assert_normals_unit_norm_on_valid(
    sample: DenseSample, *, atol: float = 1e-5
) -> None:
    raise_if_violated(check_normals_unit_norm_on_valid(sample, atol=atol))


def check_normals_camera_frame_convention(sample: DenseSample) -> InvariantReport:
    """Normals carry shape ``[3, H, W]`` in the project-wide right-down-forward frame.

    Shape + dtype check at the type boundary; the convention itself is
    declared by ADR-0001 and upheld by every transform author.
    """
    name = "normals.camera_frame_convention"
    n = require(sample.normals, "sample must carry normals")
    if n.ndim != 3 or n.size(0) != 3:
        return InvariantReport(
            name=name,
            ok=False,
            message=f"normals must be [3, H, W]; got shape {tuple(n.shape)}",
            details={"shape": str(tuple(n.shape))},
        )
    if not n.is_floating_point():
        return InvariantReport(
            name=name,
            ok=False,
            message=f"normals must be a float tensor; got dtype {n.dtype}",
            details={"dtype": str(n.dtype)},
        )
    return InvariantReport(name=name, ok=True)


def assert_normals_camera_frame_convention(sample: DenseSample) -> None:
    raise_if_violated(check_normals_camera_frame_convention(sample))
