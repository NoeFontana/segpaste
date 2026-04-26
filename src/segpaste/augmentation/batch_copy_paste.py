"""GPU-resident batched copy-paste (ADR-0008).

Public entry point replacing :class:`CopyPasteCollator` and the four
CPU wrappers (:class:`InstancePaste`, :class:`PanopticPaste`,
:class:`DepthAwarePaste`, :class:`ClassMix`). Operates entirely on
leading-batch tensors so the forward pass traces cleanly under
``torch.compile(fullgraph=True)`` — see ``tests/test_compile_clean.py``.

Pipeline (ADR-0008 §architecture):

1. :class:`BatchedPlacementSampler` draws one affine per target
   (``source_idx``, ``scale``, ``translate``, ``hflip``) under a
   diagonal-masked multinomial so every target picks a source ``!= i``.
2. :class:`AffinePropagator` applies the sampled affine to every channel
   group of the selected source via ``grid_sample`` — bilinear for
   continuous modalities (image, depth, normals), nearest for label
   modalities (instance/semantic/panoptic maps, depth_valid).
3. A union of the warped instance masks gated by ``paste_valid`` forms
   the per-batch paste mask; the z-test in :class:`TileCompositor`
   upgrades this to the effective composite mask (ADR-0005 §3).
4. :class:`TileCompositor` performs the pixelwise where-composite in
   fixed-size tiles so activation memory scales with ``tile_size²``.

Instance-row merging policy: target rows are preserved at their
original slots; pasted source rows are compacted into the target's
*free* slots (those where ``instance_valid`` was ``False``) in
source-slot order. Surplus pastes are dropped when the target has
no room — callers that require more headroom should widen ``K`` via
:meth:`BatchedDenseSample.to_padded`.
"""

from __future__ import annotations

from dataclasses import replace

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor, nn

from segpaste._internal.gpu.affine_propagate import AffinePropagator
from segpaste._internal.gpu.batched_placement import (
    BatchedPlacement,
    BatchedPlacementConfig,
    BatchedPlacementSampler,
)
from segpaste._internal.gpu.tile_composite import TileCompositor, TileCompositorConfig
from segpaste.types import PaddedBatchedDenseSample


class BatchCopyPasteConfig(BaseModel):
    """Configuration for :class:`BatchCopyPaste`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    placement: BatchedPlacementConfig = Field(default_factory=BatchedPlacementConfig)
    """Parameters for :class:`BatchedPlacementSampler`."""

    composite: TileCompositorConfig = Field(default_factory=TileCompositorConfig)
    """Parameters for :class:`TileCompositor`."""

    min_residual_area_frac: float = Field(default=0.1, ge=0.0, le=1.0)
    """Drop a target instance when its survivor mask retains less than this
    fraction of its original area. ``0.0`` keeps every target row regardless
    of occlusion (legacy behavior); ``0.1`` matches the COCO eval ergonomics
    of dropping ≥90%-occluded annotations. Mirrors the inverse of
    :attr:`segpaste._internal.composite.CompositeConfig.occluded_area_threshold`
    (``min_residual = 1 - occluded_threshold``)."""


def drop_occluded_targets(
    padded: PaddedBatchedDenseSample,
    composited: PaddedBatchedDenseSample,
    min_residual_area_frac: float,
) -> PaddedBatchedDenseSample:
    """Flip ``instance_valid`` to False for over-covered target rows.

    Compares per-instance original area (``padded.instance_masks``) to
    survivor area (``composited.instance_masks``, post tile-composite).
    Rows with ``survivor / original < min_residual_area_frac`` are dropped.
    """
    if (
        min_residual_area_frac <= 0.0
        or padded.instance_masks is None
        or composited.instance_masks is None
    ):
        return composited
    orig = padded.instance_masks.flatten(2).sum(-1).to(torch.float32)
    surv = composited.instance_masks.flatten(2).sum(-1).to(torch.float32)
    keep = surv >= min_residual_area_frac * orig
    return replace(composited, instance_valid=composited.instance_valid & keep)


class BatchCopyPaste(nn.Module):
    """Graph-compilable batched copy-paste augmentation."""

    config: BatchCopyPasteConfig

    def __init__(self, config: BatchCopyPasteConfig | None = None) -> None:
        super().__init__()
        self.config = config or BatchCopyPasteConfig()
        self.placement_sampler = BatchedPlacementSampler(self.config.placement)
        self.propagator = AffinePropagator()
        self.compositor = TileCompositor(self.config.composite)

    def forward(
        self,
        padded: PaddedBatchedDenseSample,
        generator: torch.Generator | None = None,
    ) -> PaddedBatchedDenseSample:
        if padded.batch_size == 0:
            return padded

        valid_extent = self._valid_extent(padded)
        placement = self.placement_sampler(padded, generator, valid_extent=valid_extent)
        warped = self.propagator(padded, placement)
        paste_mask = self._paste_mask(warped, placement)
        composited = self.compositor(padded, warped, paste_mask)
        composited = drop_occluded_targets(
            padded, composited, self.config.min_residual_area_frac
        )
        return self._merge_slots(composited, warped, placement)

    @staticmethod
    def _valid_extent(padded: PaddedBatchedDenseSample) -> Tensor | None:
        """Per-sample ``[B, 2]`` (h_v, w_v) bound on the unpadded image rect.

        Assumes the LSJ convention of a top-left valid rect with bottom/right
        zero-pad (:class:`FixedSizeCrop` via ``augmentation/lsj.py``).
        """
        if padded.padding_mask is None:
            return None
        not_pad = (~padded.padding_mask.as_subclass(Tensor)).squeeze(1)
        h_v = not_pad.any(dim=-1).sum(dim=-1).to(torch.float32)
        w_v = not_pad.any(dim=-2).sum(dim=-1).to(torch.float32)
        return torch.stack([h_v, w_v], dim=-1)

    @staticmethod
    def _paste_mask(
        warped: PaddedBatchedDenseSample, placement: BatchedPlacement
    ) -> Tensor:
        """Union of warped instance masks gated by ``paste_valid``.

        Returns a ``[B, H, W]`` bool tensor identifying pixels that any
        valid pasted slot contributes to.
        """
        b = warped.batch_size
        h, w = warped.images.shape[-2:]
        if warped.instance_masks is None:
            return torch.zeros((b, h, w), dtype=torch.bool, device=warped.images.device)
        gate = placement.paste_valid.view(
            placement.paste_valid.shape[0], placement.paste_valid.shape[1], 1, 1
        )
        return (warped.instance_masks & gate).any(dim=1)

    @staticmethod
    def _merge_slots(
        composited: PaddedBatchedDenseSample,
        warped: PaddedBatchedDenseSample,
        placement: BatchedPlacement,
    ) -> PaddedBatchedDenseSample:
        """Compact paste rows into the target's free slots.

        Output slot ``t`` carries the survivor-updated target row when
        ``composited.instance_valid[t]`` is ``True``, otherwise it
        receives the next pasted source row in source-slot order. When
        the number of pastes exceeds the target's free slot count, the
        surplus is dropped. The remap is computed via a ``[B, K, K]``
        rank-equality match — graph-clean and inexpensive at COCO scale.
        """
        pv = placement.paste_valid  # [B, K]
        free = ~composited.instance_valid  # [B, K]

        # Pair the n-th free target slot with the n-th source paste slot via
        # rank-equality. Ranks are unique per row, so each match row has
        # at most one True; argmax then names the source slot to gather.
        free_rank = free.long().cumsum(-1) - 1
        paste_rank = pv.long().cumsum(-1) - 1
        match = (
            free.unsqueeze(-1)
            & pv.unsqueeze(-2)
            & (free_rank.unsqueeze(-1) == paste_rank.unsqueeze(-2))
        )  # [B, K_t, K_s]
        receives = match.any(dim=-1)  # [B, K]
        # argmax returns 0 where no match — guarded by `receives` in `where`.
        src_k = match.long().argmax(dim=-1)

        b, k = pv.shape
        batch_idx = torch.arange(b, device=pv.device).unsqueeze(-1).expand(b, k)

        def gather(src: Tensor, dst: Tensor) -> Tensor:
            sel = receives.view(b, k, *([1] * (src.ndim - 2)))
            return torch.where(sel, src[batch_idx, src_k], dst)

        merged_boxes = gather(warped.boxes, composited.boxes)
        merged_labels = gather(warped.labels, composited.labels)
        merged_masks = (
            gather(warped.instance_masks, composited.instance_masks)
            if warped.instance_masks is not None
            and composited.instance_masks is not None
            else composited.instance_masks
        )
        merged_ids = (
            gather(warped.instance_ids, composited.instance_ids)
            if warped.instance_ids is not None and composited.instance_ids is not None
            else composited.instance_ids
        )
        merged_valid = composited.instance_valid | receives

        return PaddedBatchedDenseSample(
            images=composited.images,
            boxes=merged_boxes,
            labels=merged_labels,
            instance_valid=merged_valid,
            max_instances=composited.max_instances,
            instance_masks=merged_masks,
            instance_ids=merged_ids,
            semantic_maps=composited.semantic_maps,
            panoptic_maps=composited.panoptic_maps,
            depth=composited.depth,
            depth_valid=composited.depth_valid,
            normals=composited.normals,
            padding_mask=composited.padding_mask,
            camera_intrinsics=composited.camera_intrinsics,
        )
