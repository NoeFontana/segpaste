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
from torchvision import tv_tensors

from segpaste._internal.gpu.affine_propagate import AffinePropagator
from segpaste._internal.gpu.batched_placement import (
    BatchedPlacement,
    BatchedPlacementConfig,
    BatchedPlacementSampler,
)
from segpaste._internal.gpu.tile_composite import TileCompositor, TileCompositorConfig
from segpaste.types import PaddedBatchedDenseSample, PanopticSchemaSpec
from segpaste.types.dense_sample import PanopticMap, SemanticMap


class PanopticPasteConfig(BaseModel):
    """Panoptic-specific augmentation knobs (ADR-0006).

    Activates the thing-only paste source filter (ADR-0006 §2) and the
    stuff-area-threshold post-composite revert. Set on
    :attr:`BatchCopyPasteConfig.panoptic` to engage the panoptic path.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    taxonomy: PanopticSchemaSpec
    """Stuff/thing taxonomy. Required when panoptic mode is active."""

    tau_stuff_frac: float = Field(default=0.1, ge=0.0, le=1.0)
    """Per-stuff-class minimum *remaining-area fraction* after paste. A
    paste that drives any stuff class below this fraction of its
    pre-paste area is reverted on the affected pixels. ``0.1`` matches
    :attr:`BatchCopyPasteConfig.min_residual_area_frac` ergonomics."""


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

    panoptic: PanopticPasteConfig | None = None
    """When set, gates source rows to thing-only and applies the
    stuff-area-threshold post-composite revert (ADR-0006). ``None``
    leaves the augmentation panoptic-agnostic (default)."""


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
    thing_classes: Tensor
    stuff_classes: Tensor

    def __init__(self, config: BatchCopyPasteConfig | None = None) -> None:
        super().__init__()
        self.config = config or BatchCopyPasteConfig()
        self.placement_sampler = BatchedPlacementSampler(self.config.placement)
        self.propagator = AffinePropagator()
        self.compositor = TileCompositor(self.config.composite)

        if self.config.panoptic is not None:
            taxonomy = self.config.panoptic.taxonomy
            things = sorted(
                cls for cls, kind in taxonomy.classes.items() if kind == "thing"
            )
            stuffs = sorted(
                cls for cls, kind in taxonomy.classes.items() if kind == "stuff"
            )
            self.register_buffer(
                "thing_classes", torch.tensor(things, dtype=torch.int64)
            )
            self.register_buffer(
                "stuff_classes", torch.tensor(stuffs, dtype=torch.int64)
            )
            self._class_table_size = max([*things, *stuffs, taxonomy.ignore_index]) + 1
        else:
            self.register_buffer("thing_classes", torch.empty((0,), dtype=torch.int64))
            self.register_buffer("stuff_classes", torch.empty((0,), dtype=torch.int64))
            self._class_table_size = 0

    def forward(
        self,
        padded: PaddedBatchedDenseSample,
        generator: torch.Generator | None = None,
    ) -> PaddedBatchedDenseSample:
        if padded.batch_size == 0:
            return padded

        valid_extent = self._valid_extent(padded)
        source_eligible = self._source_eligible(padded)
        placement = self.placement_sampler(
            padded,
            generator,
            valid_extent=valid_extent,
            source_eligible=source_eligible,
        )
        warped = self.propagator(padded, placement)
        paste_mask = self._paste_mask(warped, placement)
        composited = self.compositor(padded, warped, paste_mask)
        if self.config.panoptic is not None:
            composited, warped = self._revert_stuff_collapse(
                padded, composited, warped, paste_mask
            )
        composited = drop_occluded_targets(
            padded, composited, self.config.min_residual_area_frac
        )
        return self._merge_slots(composited, warped, placement)

    def _source_eligible(self, padded: PaddedBatchedDenseSample) -> Tensor | None:
        if self.config.panoptic is None:
            return None
        return torch.isin(padded.labels, self.thing_classes, assume_unique=True)

    def _revert_stuff_collapse(
        self,
        padded: PaddedBatchedDenseSample,
        composited: PaddedBatchedDenseSample,
        warped: PaddedBatchedDenseSample,
        paste_mask: Tensor,
    ) -> tuple[PaddedBatchedDenseSample, PaddedBatchedDenseSample]:
        """Revert paste pixels where a pre-paste stuff class collapsed (ADR-0006 §3).

        Image, semantic, and panoptic modalities are reverted to ``padded`` on
        pixels that (a) carried a stuff class pre-paste whose post-paste area
        fell below ``tau_stuff_frac`` of its pre-paste area, and (b) were
        overwritten by paste. Target instance survivors are restored and
        warped paste masks cleared so the panoptic bijection on thing pixels
        still holds downstream.
        """
        panoptic = self.config.panoptic
        if (
            panoptic is None
            or padded.semantic_maps is None
            or composited.semantic_maps is None
            or self.stuff_classes.numel() == 0
        ):
            return composited, warped

        b, h, w = paste_mask.shape
        device = paste_mask.device
        n = self._class_table_size
        tau = panoptic.tau_stuff_frac

        pre = padded.semantic_maps.as_subclass(Tensor).flatten(1)
        post = composited.semantic_maps.as_subclass(Tensor).flatten(1)
        ones = torch.ones((), dtype=torch.int64, device=device).expand_as(pre)
        before_hist = torch.zeros((b, n), dtype=torch.int64, device=device)
        after_hist = torch.zeros((b, n), dtype=torch.int64, device=device)
        before_hist.scatter_add_(1, pre, ones)
        after_hist.scatter_add_(1, post, ones)

        before = before_hist.index_select(1, self.stuff_classes).to(torch.float32)
        after = after_hist.index_select(1, self.stuff_classes).to(torch.float32)
        collapse = (before > 0) & (after < tau * before)

        collapse_table = torch.zeros((b, n), dtype=torch.bool, device=device)
        collapse_table.scatter_(
            1, self.stuff_classes.unsqueeze(0).expand(b, -1), collapse
        )
        revert = collapse_table.gather(1, pre).view(b, h, w) & paste_mask

        rev3 = revert.unsqueeze(1)
        new_image = torch.where(rev3, padded.images, composited.images)
        new_sem = torch.where(
            revert,
            padded.semantic_maps.as_subclass(Tensor),
            composited.semantic_maps.as_subclass(Tensor),
        )
        if padded.panoptic_maps is not None and composited.panoptic_maps is not None:
            new_pano: Tensor | None = torch.where(
                revert,
                padded.panoptic_maps.as_subclass(Tensor),
                composited.panoptic_maps.as_subclass(Tensor),
            )
        else:
            new_pano = None

        if composited.instance_masks is not None and padded.instance_masks is not None:
            new_target_masks: Tensor | None = composited.instance_masks | (
                padded.instance_masks & rev3
            )
        else:
            new_target_masks = composited.instance_masks

        new_warped_masks = (
            warped.instance_masks & ~rev3 if warped.instance_masks is not None else None
        )

        new_composited = replace(
            composited,
            images=tv_tensors.Image(new_image),
            semantic_maps=SemanticMap(new_sem),
            panoptic_maps=PanopticMap(new_pano) if new_pano is not None else None,
            instance_masks=new_target_masks,
        )
        new_warped = replace(warped, instance_masks=new_warped_masks)
        return new_composited, new_warped

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
