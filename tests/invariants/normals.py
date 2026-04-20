"""Normals-modality invariants per ADR-0001 §(ii)."""

import torch

from segpaste.types import DenseSample
from tests.invariants import require


def assert_normals_unit_norm_on_valid(
    sample: DenseSample, *, atol: float = 1e-5
) -> None:
    """``||n(p)||_2 == 1`` on every valid pixel within ``atol``."""
    normals = require(sample.normals, "sample must carry normals")
    norms = normals.norm(dim=0)
    if sample.depth_valid is not None:
        valid = sample.depth_valid.squeeze(0).to(torch.bool)
    else:
        valid = torch.ones_like(norms, dtype=torch.bool)
    if not bool(valid.any()):
        return
    err = (norms[valid] - 1.0).abs()
    if bool((err > atol).any()):
        raise AssertionError(
            f"normals violate unit norm (max error {err.max().item():.3e})"
        )


def assert_normals_camera_frame_convention(sample: DenseSample) -> None:
    """Normals carry shape ``[3, H, W]`` in the project-wide right-down-forward frame.

    Shape + dtype check at the type boundary; the convention itself is
    declared by ADR-0001 and upheld by every transform author.
    """
    n = require(sample.normals, "sample must carry normals")
    if n.ndim != 3 or n.size(0) != 3:
        raise AssertionError(f"normals must be [3, H, W]; got shape {tuple(n.shape)}")
    if not n.is_floating_point():
        raise AssertionError(f"normals must be a float tensor; got dtype {n.dtype}")
