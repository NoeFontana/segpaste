"""Vectorized per-sample placement sampler (ADR-0008).

Replaces the per-instance Python loop inside
:class:`segpaste._internal.placement.PlacementSampler` with a single batched
kernel. Emits per-target affine parameters ``(source_idx, translate, scale,
hflip)`` and a per-slot validity gate consumed by
:mod:`segpaste._internal.gpu.affine_propagate`.

All operations are leading-batch tensor ops — no ``.item()``, no Python loops
over ``B`` or ``K`` — so the module traces cleanly under
``torch.compile(fullgraph=True)``.

Intra-batch source selection uses a diagonal-masked ``torch.multinomial``
draw so every target picks a source ``j != i``. ``B == 1`` is a degenerate
batch with no valid source and yields ``paste_valid = False`` everywhere;
callers should gate on ``batch_size > 1`` if pastes are required.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor, nn

from segpaste.types import PaddedBatchedDenseSample


class BatchedPlacementConfig(BaseModel):
    """Configuration for :class:`BatchedPlacementSampler`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scale_range: tuple[float, float] = (0.5, 2.0)
    """Per-target isotropic scale drawn uniformly from this range."""

    hflip_probability: float = Field(default=0.5, ge=0.0, le=1.0)
    """Per-target probability of horizontal flip."""

    paste_prob: float = Field(default=1.0, ge=0.0, le=1.0)
    """Per-image Bernoulli gate on whether the augmentation fires. ``1.0``
    pastes every image (current behavior); ``0.5`` pastes half the batch."""

    k_range: tuple[int, int] = (1, 256)
    """Per-image paste-count cap. ``K_max`` is drawn uniformly from
    ``[k_range[0], k_range[1]]`` and any slot whose cumulative rank in
    ``paste_valid`` exceeds it is dropped. Default ``(1, 256)`` is a no-op
    for any sane ``max_instances`` (matches the panoptic schema default)."""


@dataclass(frozen=True, slots=True)
class BatchedPlacement:
    """Per-target affine placement parameters.

    ``source_idx[i]`` names the source sample index for target ``i`` (always
    ``!= i`` when ``B > 1``). ``translate`` is in pixel units of the image
    frame — the source's top-left corner after translation. ``paste_valid``
    gates per-slot paste writes: ``False`` for slots whose source row is
    invalid or whose scaled bounding box would lie entirely outside the
    image frame after the affine is applied.
    """

    source_idx: Tensor  # [B] int64
    translate: Tensor  # [B, 2] float32 (ty, tx)
    scale: Tensor  # [B] float32 isotropic
    hflip: Tensor  # [B] bool
    paste_valid: Tensor  # [B, K] bool


class BatchedPlacementSampler(nn.Module):
    """Vectorized per-target placement sampler.

    Scope note: this replaces the CPU :class:`PlacementSampler` in a
    *statistically* equivalent sense (ADR-0008 §KS), not a pixel-exact sense.
    The CPU sampler does per-instance placement under an IoU-collision
    retry loop; this sampler draws one affine per target without per-slot
    collision checking. Divergence is bounded by the KS soft-gate over
    paste-area, per-image paste count, and per-class histograms — see
    ``tests/test_batch_copy_paste_ks.py``.
    """

    config: BatchedPlacementConfig

    def __init__(self, config: BatchedPlacementConfig | None = None) -> None:
        super().__init__()
        self.config = config or BatchedPlacementConfig()

    def forward(
        self,
        padded: PaddedBatchedDenseSample,
        generator: torch.Generator | None = None,
        valid_extent: Tensor | None = None,
        source_eligible: Tensor | None = None,
    ) -> BatchedPlacement:
        """Draw one affine per target.

        ``valid_extent`` is an optional ``[B, 2]`` ``(h_v, w_v)`` float tensor
        bounding the valid (non-pad) image extent for each sample. When set,
        translates are sampled inside ``[0, h_v) x [0, w_v)`` and pastes whose
        source bbox extends past the source's valid extent are dropped via
        ``paste_valid``. ``None`` recovers the full-canvas behavior.

        ``source_eligible`` is an optional ``[B, K]`` bool tensor gating which
        rows are eligible as paste *sources*. Used by the panoptic preset to
        restrict paste sources to thing-class rows (ADR-0006 §2). ``None``
        treats every valid row as eligible.
        """
        b = padded.batch_size
        k = padded.max_instances
        device = padded.images.device
        h, w = padded.images.shape[-2:]

        if b == 0:
            return BatchedPlacement(
                source_idx=torch.empty((0,), dtype=torch.int64, device=device),
                translate=torch.empty((0, 2), dtype=torch.float32, device=device),
                scale=torch.empty((0,), dtype=torch.float32, device=device),
                hflip=torch.empty((0,), dtype=torch.bool, device=device),
                paste_valid=torch.empty((0, k), dtype=torch.bool, device=device),
            )

        if b == 1:
            return BatchedPlacement(
                source_idx=torch.zeros((1,), dtype=torch.int64, device=device),
                translate=torch.zeros((1, 2), dtype=torch.float32, device=device),
                scale=torch.ones((1,), dtype=torch.float32, device=device),
                hflip=torch.zeros((1,), dtype=torch.bool, device=device),
                paste_valid=torch.zeros((1, k), dtype=torch.bool, device=device),
            )

        if valid_extent is None:
            ve = padded.images.new_tensor([float(h), float(w)]).expand(b, 2)
        else:
            ve = valid_extent.to(device=device, dtype=torch.float32)
        tgt_h = ve[:, 0]
        tgt_w = ve[:, 1]

        weights = 1.0 - torch.eye(b, device=device)
        source_idx = torch.multinomial(
            weights, num_samples=1, generator=generator
        ).squeeze(-1)

        source_boxes = padded.boxes[source_idx]
        source_valid = padded.instance_valid[source_idx]
        if source_eligible is not None:
            source_valid = source_valid & source_eligible[source_idx]

        smin, smax = self.config.scale_range
        scale = (
            torch.rand((b,), generator=generator, device=device) * (smax - smin) + smin
        )

        hflip = (
            torch.empty((b,), device=device, dtype=torch.float32)
            .bernoulli_(self.config.hflip_probability, generator=generator)
            .bool()
        )

        # Per-slot effective right/bottom edges in the source coord frame.
        # Affine maps source (sy, sx) → output (sy*s + ty, sx*s + tx); with hflip
        # x is reflected about (w-1)/2 first, so the box's effective right edge
        # in output coords is ((w-1) - bx1)*s + tx (vs. bx2*s + tx with no flip).
        flip = hflip.unsqueeze(-1)
        eff_x2 = torch.where(
            flip, (w - 1.0) - source_boxes[..., 0], source_boxes[..., 2]
        )
        eff_y2 = source_boxes[..., 3]
        # Padded slots (boxes [0,0,0,0]) pick up eff_x2 = w-1 under hflip — large
        # and bogus. Zero them out so they don't over-restrict the per-target
        # translate bound; source_valid still gates them in `paste_valid`.
        zero = torch.zeros_like(eff_x2)
        eff_x2_for_bound = torch.where(source_valid, eff_x2, zero)
        eff_y2_for_bound = torch.where(source_valid, eff_y2, zero)

        scaled_eff_x2 = eff_x2 * scale.unsqueeze(-1)
        scaled_eff_y2 = eff_y2 * scale.unsqueeze(-1)
        max_scaled_x2 = (eff_x2_for_bound * scale.unsqueeze(-1)).amax(dim=-1)
        max_scaled_y2 = (eff_y2_for_bound * scale.unsqueeze(-1)).amax(dim=-1)

        max_ty = torch.clamp(tgt_h - max_scaled_y2, min=0.0)
        max_tx = torch.clamp(tgt_w - max_scaled_x2, min=0.0)
        ty = torch.rand((b,), generator=generator, device=device) * max_ty
        tx = torch.rand((b,), generator=generator, device=device) * max_tx
        translate = torch.stack([ty, tx], dim=-1)

        fits = (scaled_eff_y2 + ty.unsqueeze(-1) <= tgt_h.unsqueeze(-1)) & (
            scaled_eff_x2 + tx.unsqueeze(-1) <= tgt_w.unsqueeze(-1)
        )
        source_ve = ve[source_idx]
        box_in_valid = (source_boxes[..., 3] <= source_ve[:, 0:1]) & (
            source_boxes[..., 2] <= source_ve[:, 1:2]
        )
        paste_valid = source_valid & fits & box_in_valid

        do_paste = (
            torch.empty((b,), device=device, dtype=torch.float32)
            .bernoulli_(self.config.paste_prob, generator=generator)
            .bool()
        )
        paste_valid = paste_valid & do_paste.unsqueeze(-1)

        k_lo, k_hi = self.config.k_range
        k_max = torch.randint(
            low=k_lo,
            high=k_hi + 1,
            size=(b,),
            device=device,
            generator=generator,
        )
        rank = paste_valid.long().cumsum(dim=-1) - 1
        paste_valid = paste_valid & (rank < k_max.unsqueeze(-1))

        return BatchedPlacement(
            source_idx=source_idx,
            translate=translate,
            scale=scale,
            hflip=hflip,
            paste_valid=paste_valid,
        )
