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

Instance-row merging policy: output slot ``k`` carries the pasted row
when ``paste_valid[b, k]`` is true, and the survivor-updated target
row otherwise. Overflow beyond ``max_instances`` is not attempted —
callers that require more capacity should widen ``K`` via
:meth:`BatchedDenseSample.to_padded`.
"""

from __future__ import annotations

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

        placement = self.placement_sampler(padded, generator)
        warped = self.propagator(padded, placement)
        paste_mask = self._paste_mask(warped, placement)
        composited = self.compositor(padded, warped, paste_mask)
        return self._merge_slots(composited, warped, placement)

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
        """Per-slot merge under ``paste_valid``.

        Slots where ``paste_valid`` is ``True`` receive the warped/pasted
        row; the remaining slots keep the composited survivor row.
        """
        pv = placement.paste_valid  # [B, K]
        pv2 = pv.unsqueeze(-1)  # [B, K, 1]
        pv4 = pv.view(pv.shape[0], pv.shape[1], 1, 1)  # [B, K, 1, 1]

        merged_masks = composited.instance_masks
        if composited.instance_masks is not None and warped.instance_masks is not None:
            merged_masks = torch.where(
                pv4, warped.instance_masks, composited.instance_masks
            )

        merged_ids = composited.instance_ids
        if composited.instance_ids is not None and warped.instance_ids is not None:
            merged_ids = torch.where(pv, warped.instance_ids, composited.instance_ids)

        merged_boxes = torch.where(pv2, warped.boxes, composited.boxes)
        merged_labels = torch.where(pv, warped.labels, composited.labels)
        merged_valid = composited.instance_valid | pv

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
