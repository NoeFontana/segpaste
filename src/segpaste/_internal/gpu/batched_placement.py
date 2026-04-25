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
    ) -> BatchedPlacement:
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

        weights = 1.0 - torch.eye(b, device=device)
        source_idx = torch.multinomial(
            weights, num_samples=1, generator=generator
        ).squeeze(-1)

        source_boxes = padded.boxes[source_idx]
        source_valid = padded.instance_valid[source_idx]

        smin, smax = self.config.scale_range
        scale = (
            torch.rand((b,), generator=generator, device=device) * (smax - smin) + smin
        )

        box_wh = source_boxes[..., 2:] - source_boxes[..., :2]
        scaled_w_per_slot = box_wh[..., 0] * scale.unsqueeze(-1)
        scaled_h_per_slot = box_wh[..., 1] * scale.unsqueeze(-1)

        max_scaled_h = scaled_h_per_slot.amax(dim=-1)
        max_scaled_w = scaled_w_per_slot.amax(dim=-1)
        max_ty = torch.clamp(float(h) - max_scaled_h, min=0.0)
        max_tx = torch.clamp(float(w) - max_scaled_w, min=0.0)
        ty = torch.rand((b,), generator=generator, device=device) * max_ty
        tx = torch.rand((b,), generator=generator, device=device) * max_tx
        translate = torch.stack([ty, tx], dim=-1)

        hflip = torch.bernoulli(
            torch.full(
                (b,),
                self.config.hflip_probability,
                device=device,
                dtype=torch.float32,
            ),
            generator=generator,
        ).bool()

        fits = (scaled_h_per_slot <= float(h)) & (scaled_w_per_slot <= float(w))
        paste_valid = source_valid & fits

        return BatchedPlacement(
            source_idx=source_idx,
            translate=translate,
            scale=scale,
            hflip=hflip,
            paste_valid=paste_valid,
        )
