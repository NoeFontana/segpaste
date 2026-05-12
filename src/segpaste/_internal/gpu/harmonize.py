"""Image harmonization for batched copy-paste (ADR-0012).

Operates on the *full warped image* between :class:`AffinePropagator` and
:class:`TileCompositor` in :class:`BatchCopyPaste.forward`. Three pure-torch
modes — Reinhard 2001 statistical color transfer, Burt & Adelson 1983
multi-band pyramid blending, and a Poisson-equation gradient-domain solve
diagonalized by the type-I Discrete Sine Transform. All modes stay inside
``torch.compile(fullgraph=True)`` so the empty
``scripts/compile_allowlist.txt`` is preserved.

A per-image ``bernoulli(prob)`` draw chooses, for each sample independently,
whether to use the harmonized image or the un-harmonized warped image. Both
arms always run when ``prob > 0`` — the curriculum cost is the cost the user
accepts to keep the model exposed to both composites.

The compositor downstream still applies the z-test alpha-where over the
result, so harmonized content for pixels rejected by the depth z-test (or
outside ``paste_mask``) is computed and unused. That waste is intentional:
keeping the harmonization global lets multi-band pyramids and DST solves
have the support they need without seam artifacts.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Literal

import torch
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor, nn
from torchvision import tv_tensors

from segpaste.types import PaddedBatchedDenseSample


class HarmonizeConfig(BaseModel):
    """Configuration for :class:`ImageHarmonizer` (ADR-0012)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: Literal["reinhard", "multiband", "poisson"] = "multiband"
    """Which harmonization mode to apply when the bernoulli draw fires.
    ``"multiband"`` (Burt-Adelson 1983) is the recommended default — captures
    most of what `cv2.seamlessClone` provides perceptually at a fraction of
    the implementation surface."""

    prob: float = Field(default=0.0, ge=0.0, le=1.0)
    """Per-image probability that harmonization is applied. Default ``0.0``
    is a graph-clean fast path that returns the un-harmonized warped image
    unchanged — bitwise-identical to v0.3.0 behavior. Values in ``(0, 1)``
    expose the model to both harmonized and un-harmonized composites
    (curriculum). ``1.0`` always harmonizes."""

    pyramid_levels: int = Field(default=5, gt=0)
    """Number of pyramid levels for ``"multiband"`` mode. The runtime cap is
    ``floor(log2(min(H, W))) - 1`` so the smallest level still has at least
    4 pixels per side."""


# Reinhard 2001 (Color Transfer between Images, IEEE CG&A) -- RGB <-> LMS
# <-> L-alpha-beta matrices. The L-alpha-beta space is the perceptually-
# decorrelated logarithmic LMS basis introduced in section 3 of the paper;
# statistics matched in this space avoid the cross-channel coupling that
# biases naive RGB mean/std matching.

_RGB_TO_LMS = torch.tensor(
    [
        [0.3811, 0.5783, 0.0402],
        [0.1967, 0.7244, 0.0782],
        [0.0241, 0.1288, 0.8444],
    ],
    dtype=torch.float32,
)

# diag(1/sqrt(3), 1/sqrt(6), 1/sqrt(2)) @ [[1,1,1],[1,1,-2],[1,-1,0]] applied
# after log10.
_LMS_TO_LAB = torch.tensor(
    [
        [1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0)],
        [1.0 / math.sqrt(6.0), 1.0 / math.sqrt(6.0), -2.0 / math.sqrt(6.0)],
        [1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0), 0.0],
    ],
    dtype=torch.float32,
)


def _binomial_kernel_5() -> Tensor:
    """5x5 binomial kernel: outer product of ``[1, 4, 6, 4, 1] / 16``.

    Standard low-pass for Burt-Adelson Gaussian pyramids; symmetric, separable,
    and the closest 5-tap integer approximation to a unit-stddev Gaussian.
    """
    k = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0
    return torch.outer(k, k)  # [5, 5]


def _dst_matrix(n: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Orthonormal DST-I matrix of size ``n x n``.

    ``M[i, j] = sqrt(2/(n+1)) * sin(pi * (i+1) * (j+1) / (n+1))`` for
    0-indexed ``(i, j)``. The DST-I is its own inverse for the orthonormal
    scaling.
    """
    idx = torch.arange(1, n + 1, device=device, dtype=dtype)
    return math.sqrt(2.0 / (n + 1)) * torch.sin(
        math.pi * idx.unsqueeze(1) * idx.unsqueeze(0) / (n + 1)
    )


def _dst_eigenvalues(
    h: int, w: int, device: torch.device, dtype: torch.dtype
) -> Tensor:
    """``[H, W]`` eigenvalues of ``-Laplacian`` under DST-I diagonalization.

    The 1-D discrete Laplacian with Dirichlet boundary has eigenvalues
    ``lam_i = -4 sin^2(pi * i / (2 * (N + 1)))``. We return the absolute
    value (strictly positive) so the Poisson solve divides by a strictly
    positive number.
    """
    i = torch.arange(1, h + 1, device=device, dtype=dtype) / (h + 1)
    j = torch.arange(1, w + 1, device=device, dtype=dtype) / (w + 1)
    lam_h = 4.0 * torch.sin(math.pi * i / 2.0) ** 2  # [H]
    lam_w = 4.0 * torch.sin(math.pi * j / 2.0) ** 2  # [W]
    return lam_h.unsqueeze(1) + lam_w.unsqueeze(0)  # [H, W]


_LAPLACIAN_STENCIL = torch.tensor(
    [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32
)


class ImageHarmonizer(nn.Module):
    """Per-image image harmonization between propagator and compositor.

    ``forward`` returns a :class:`PaddedBatchedDenseSample` mirroring
    ``warped`` with only the ``images`` field replaced. All other modalities
    pass through unchanged — harmonization is image-channel only by design,
    in line with ADR-0007 §7's restriction of label modalities to nearest-
    sample composites.

    When ``config.prob == 0`` the call is a fast-path identity: the warped
    sample is returned unchanged with no compute and no graph branches taken.
    The mode dispatch (``if/elif`` on a frozen Pydantic literal) is resolved
    at trace time so ``torch.compile`` specializes one branch per module
    instance.
    """

    config: HarmonizeConfig

    def __init__(self, config: HarmonizeConfig | None = None) -> None:
        super().__init__()
        self.config = config or HarmonizeConfig()

        self.register_buffer(
            "_binom", _binomial_kernel_5().view(1, 1, 5, 5), persistent=False
        )
        self.register_buffer("_rgb_to_lms", _RGB_TO_LMS.clone(), persistent=False)
        self.register_buffer(
            "_lms_to_rgb", torch.linalg.inv(_RGB_TO_LMS), persistent=False
        )
        self.register_buffer("_lms_to_lab", _LMS_TO_LAB.clone(), persistent=False)
        self.register_buffer(
            "_laplacian_stencil",
            _LAPLACIAN_STENCIL.view(1, 1, 3, 3),
            persistent=False,
        )

    def forward(
        self,
        target: PaddedBatchedDenseSample,
        warped: PaddedBatchedDenseSample,
        paste_mask: Tensor,
        generator: torch.Generator | None = None,
    ) -> PaddedBatchedDenseSample:
        if self.config.prob == 0.0:
            return warped

        warped_img = warped.images.as_subclass(Tensor)
        target_img = target.images.as_subclass(Tensor)

        if self.config.mode == "reinhard":
            harmonized = self._reinhard(warped_img, target_img, paste_mask)
        elif self.config.mode == "multiband":
            harmonized = self._multiband(warped_img, target_img, paste_mask)
        else:  # "poisson"
            harmonized = self._poisson(warped_img, target_img, paste_mask)

        b = warped_img.size(0)
        bern = torch.bernoulli(
            torch.full(
                (b,), self.config.prob, device=warped_img.device, dtype=warped_img.dtype
            ),
            generator=generator,
        ).bool()
        out = torch.where(bern.view(b, 1, 1, 1), harmonized, warped_img)
        return replace(warped, images=tv_tensors.Image(out))

    def _reinhard(self, src: Tensor, tgt: Tensor, mask: Tensor) -> Tensor:
        """Match source crop's L-alpha-beta statistics to target background's.

        ``mu_s, sigma_s`` are computed over the warped image *inside* the
        paste mask; ``mu_t, sigma_t`` over the target image *outside* the
        paste mask (the existing scene the pasted instance is being placed
        into). Output is clamped to ``[0, 1]`` since the matched
        L-alpha-beta may lie outside the displayable RGB cube.
        """
        src_lab = self._rgb_to_lab(src)
        tgt_lab = self._rgb_to_lab(tgt)

        soft = mask.unsqueeze(1).to(src.dtype)  # [B, 1, H, W]
        s_mu, s_sigma = self._masked_mean_std(src_lab, soft)
        t_mu, t_sigma = self._masked_mean_std(tgt_lab, 1.0 - soft)

        scale = (t_sigma / s_sigma).unsqueeze(-1).unsqueeze(-1)  # [B, 3, 1, 1]
        bias = t_mu.unsqueeze(-1).unsqueeze(-1) - scale * s_mu.unsqueeze(-1).unsqueeze(
            -1
        )
        matched_lab = src_lab * scale + bias

        return self._lab_to_rgb(matched_lab).clamp(0.0, 1.0)

    @staticmethod
    def _masked_mean_std(x: Tensor, soft_mask: Tensor) -> tuple[Tensor, Tensor]:
        """Per-image, per-channel mean and std of ``x`` weighted by ``soft_mask``.

        Returns ``(mu [B, C], sigma [B, C])``. The denominator is clamped to
        ``1`` so an empty mask yields ``mu = 0`` rather than NaN; the caller
        then degrades to a zero-shift identity for that image. The variance
        is clamped to ``1e-12`` before sqrt so an all-uniform region yields
        a tiny but non-zero ``sigma`` (avoids divide-by-zero in the Reinhard
        ratio without inflating the shift).
        """
        denom = soft_mask.sum(dim=(-2, -1)).clamp(min=1.0)  # [B, 1]
        mu = (x * soft_mask).sum(dim=(-2, -1)) / denom  # [B, C]
        diff = x - mu.unsqueeze(-1).unsqueeze(-1)
        var = (diff * diff * soft_mask).sum(dim=(-2, -1)) / denom
        return mu, var.clamp(min=1e-12).sqrt()

    def _rgb_to_lab(self, rgb: Tensor) -> Tensor:
        lms = torch.einsum("cd,bdhw->bchw", self._rgb_to_lms, rgb)
        log_lms = torch.log10(lms.clamp(min=1e-6))
        return torch.einsum("cd,bdhw->bchw", self._lms_to_lab, log_lms)

    def _lab_to_rgb(self, lab: Tensor) -> Tensor:
        # _lms_to_lab is orthonormal so its inverse is its transpose; the
        # flipped einsum index applies it without buffering a second copy.
        log_lms = torch.einsum("dc,bdhw->bchw", self._lms_to_lab, lab)
        lms = torch.pow(10.0, log_lms)
        return torch.einsum("cd,bdhw->bchw", self._lms_to_rgb, lms)

    def _multiband(self, src: Tensor, tgt: Tensor, mask: Tensor) -> Tensor:
        """Burt & Adelson 1983 multi-resolution spline.

        Builds Gaussian pyramids of ``src``, ``tgt``, and ``mask.float()``;
        builds Laplacian pyramids of ``src`` and ``tgt``; blends each
        Laplacian level by the corresponding Gaussian-pyramid mask; collapses
        the blended Laplacian top-down.

        Coarse mask levels (large support) handle low-frequency
        color/illumination mismatch; fine mask levels (small support) preserve
        the silhouette's high-frequency detail. No per-image bbox handling
        and no special-casing of the mask boundary -- the pyramid does the
        seam suppression in the standard way.
        """
        soft = mask.unsqueeze(1).to(src.dtype)  # [B, 1, H, W]
        h, w = src.shape[-2], src.shape[-1]
        levels = self._effective_levels(h, w)

        src_g = self._gaussian_pyramid(src, levels)
        tgt_g = self._gaussian_pyramid(tgt, levels)
        msk_g = self._gaussian_pyramid(soft, levels)

        src_l = self._laplacian_pyramid(src_g)
        tgt_l = self._laplacian_pyramid(tgt_g)

        blended = [
            m * sl + (1.0 - m) * tl
            for sl, tl, m in zip(src_l, tgt_l, msk_g, strict=True)
        ]
        return self._reconstruct(blended).clamp(0.0, 1.0)

    def _effective_levels(self, h: int, w: int) -> int:
        """Cap pyramid depth so the smallest level has at least 4 px per side."""
        cap = max(1, int(math.log2(max(2, min(h, w)))) - 1)
        return min(self.config.pyramid_levels, cap)

    def _downsample(self, x: Tensor) -> Tensor:
        c = x.size(1)
        binom: Tensor = self._binom  # pyright: ignore[reportAssignmentType]
        kernel = binom.expand(c, 1, 5, 5).contiguous()
        return F.conv2d(x, kernel, stride=2, padding=2, groups=c)

    @staticmethod
    def _upsample_to(x: Tensor, size: tuple[int, int]) -> Tensor:
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def _gaussian_pyramid(self, x: Tensor, levels: int) -> list[Tensor]:
        pyramid = [x]
        for _ in range(levels - 1):
            pyramid.append(self._downsample(pyramid[-1]))
        return pyramid

    def _laplacian_pyramid(self, gaussian: list[Tensor]) -> list[Tensor]:
        laplacian: list[Tensor] = []
        for k in range(len(gaussian) - 1):
            up = self._upsample_to(
                gaussian[k + 1], (gaussian[k].shape[-2], gaussian[k].shape[-1])
            )
            laplacian.append(gaussian[k] - up)
        laplacian.append(gaussian[-1])
        return laplacian

    def _reconstruct(self, laplacian: list[Tensor]) -> Tensor:
        out = laplacian[-1]
        for k in range(len(laplacian) - 2, -1, -1):
            out = (
                self._upsample_to(out, (laplacian[k].shape[-2], laplacian[k].shape[-1]))
                + laplacian[k]
            )
        return out

    def _poisson(self, src: Tensor, tgt: Tensor, mask: Tensor) -> Tensor:
        """Perez et al. 2003 seamless cloning, full-image DST formulation.

        Standard discrete Poisson with Dirichlet boundary conditions on the
        image edge: solving ``-Lap(u) = -f`` in the interior with
        ``u = tgt`` on the image boundary, where
        ``f = (Lap(src) - Lap(tgt)) * mask``. Inside the paste mask this
        enforces the source's gradient field; outside the mask ``f = 0``
        so the solution relaxes back to ``tgt``.

        We work in the substitution ``u' = u - tgt`` so the boundary
        condition is homogeneous (``u' = 0`` on the image edge), which is
        what the DST-I diagonalizes. The forward and inverse DSTs are
        ``M @ x @ M.T`` with the orthonormal DST-I matrix ``M`` (its own
        inverse). Two batched matmuls per direction; cost dominated by the
        two pairs ``O(B * C * H * W * (H + W))``.

        Trade-off vs. the per-bbox formulation: the global solve may bleed
        very-low-frequency tint into the region just outside the paste mask
        (where ``f = 0`` but the boundary condition propagates). For typical
        copy-paste scales (paste rect << image), this is bounded and the
        downstream alpha-where snaps the unmasked region back to the target
        anyway.
        """
        _, c, h, w = src.shape
        device = src.device
        dtype = src.dtype

        dst_h = _dst_matrix(h, device, dtype)
        dst_w = _dst_matrix(w, device, dtype)
        eigvals = _dst_eigenvalues(h, w, device, dtype)  # [H, W], strictly positive

        stencil: Tensor = self._laplacian_stencil  # pyright: ignore[reportAssignmentType]
        kernel = stencil.expand(c, 1, 3, 3).contiguous()
        src_lap = F.conv2d(src, kernel, padding=1, groups=c)
        tgt_lap = F.conv2d(tgt, kernel, padding=1, groups=c)

        soft = mask.unsqueeze(1).to(dtype)  # [B, 1, H, W]
        f = (src_lap - tgt_lap) * soft  # [B, C, H, W]

        # 2-D DST: y = M_h @ x @ M_w.T (the two matmuls separable).
        f_dst = torch.matmul(dst_h, torch.matmul(f, dst_w.transpose(0, 1)))
        # Solve: DST{Lap(u')}_ij = lam_ij * DST{u'}_ij with lam_ij = -eigvals_ij,
        # so DST{u'}_ij = -DST{f}_ij / eigvals_ij.
        u_prime_dst = -f_dst / eigvals
        u_prime = torch.matmul(dst_h.transpose(0, 1), torch.matmul(u_prime_dst, dst_w))

        return (u_prime + tgt).clamp(0.0, 1.0)
