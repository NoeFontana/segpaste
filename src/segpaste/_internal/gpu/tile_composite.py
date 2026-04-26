"""Tiled pixelwise where-composite over :class:`PaddedBatchedDenseSample`.

ADR-0008 C5. Operates on a target/source pair already in a shared image
frame (i.e. after :class:`AffinePropagator`) and a per-batch paste mask.
Pixelwise modalities (image, semantic, panoptic, depth, depth_valid,
normals) are composited under the effective mask
``M_eff = paste_mask & z-test`` exactly as in
:class:`segpaste._internal.composite.DenseComposite` — see ADR-0005 §3.

Processing proceeds tile-by-tile with a fixed tile window so peak
activation memory scales with ``tile_size²`` rather than ``H·W``. At
``tile_size >= max(H, W)`` the iterator degenerates to a single pass
and is bitwise-equivalent to the untiled composite.

Instance-row merging (survivor + pasted concatenation, box refit) is a
BatchCopyPaste concern and is out of scope here.
"""

from __future__ import annotations

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor, nn
from torchvision import tv_tensors

from segpaste.types import PaddedBatchedDenseSample
from segpaste.types.dense_sample import PanopticMap, SemanticMap


class TileCompositorConfig(BaseModel):
    """Configuration for :class:`TileCompositor`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tile_size: int = Field(default=512, gt=0)
    """Edge length of each square tile in pixels. ``>= max(H, W)`` disables
    tiling and is bitwise-equivalent to a single-pass composite."""


def _effective_mask(
    paste_mask: Tensor,
    target_depth: Tensor | None,
    target_depth_valid: Tensor | None,
    source_depth: Tensor | None,
    source_image_valid: Tensor | None,
) -> Tensor:
    """Z-test-gated paste mask (ADR-0005 §3, ADR-0007 §2 generalization)."""
    if target_depth is None or source_depth is None:
        m_eff = paste_mask
    else:
        if target_depth_valid is None:
            raise ValueError("target depth_valid required when depth is present")
        closer = (source_depth < target_depth).squeeze(1)
        invalid = (~target_depth_valid).squeeze(1)
        m_eff = paste_mask & (closer | invalid)
    if source_image_valid is not None:
        m_eff = m_eff & source_image_valid
    return m_eff


class TileCompositor(nn.Module):
    """Tiled pixelwise where-composite for padded batched samples.

    ``forward`` returns a new :class:`PaddedBatchedDenseSample` mirroring
    ``target`` with pixelwise modalities where-composited from ``source``
    under ``paste_mask ∧ z_test``. Instance masks on the target are
    survivor-updated (``target_masks & ~m_eff``); pasted rows remain on
    ``source`` for downstream row merging.
    """

    config: TileCompositorConfig

    def __init__(self, config: TileCompositorConfig | None = None) -> None:
        super().__init__()
        self.config = config or TileCompositorConfig()

    def forward(
        self,
        target: PaddedBatchedDenseSample,
        source: PaddedBatchedDenseSample,
        paste_mask: Tensor,
    ) -> PaddedBatchedDenseSample:
        if paste_mask.dtype != torch.bool:
            raise ValueError("paste_mask must be bool")
        tgt_imgs = target.images
        b, _, h, w = tgt_imgs.shape
        if paste_mask.shape != (b, h, w):
            raise ValueError("paste_mask must have shape [B, H, W]")

        src_imgs = source.images
        tgt_sem = target.semantic_maps
        src_sem = source.semantic_maps
        tgt_pano = target.panoptic_maps
        src_pano = source.panoptic_maps

        # Pixelwise outputs: every pixel inside the tile loop gets written
        # via torch.where, so empty_like avoids the wasted clone copy.
        out_image = torch.empty_like(tgt_imgs)
        out_semantic = torch.empty_like(tgt_sem) if tgt_sem is not None else None
        out_panoptic = torch.empty_like(tgt_pano) if tgt_pano is not None else None
        out_depth = torch.empty_like(target.depth) if target.depth is not None else None
        out_depth_valid = (
            torch.empty_like(target.depth_valid)
            if target.depth_valid is not None
            else None
        )
        out_normals = (
            torch.empty_like(target.normals) if target.normals is not None else None
        )
        # Survivor masks update reads from the prior value, so we must seed
        # with target.instance_masks (clone) rather than empty_like.
        out_target_masks = (
            target.instance_masks.clone() if target.instance_masks is not None else None
        )

        src_image_valid = (
            ~source.padding_mask.as_subclass(Tensor).squeeze(1)
            if source.padding_mask is not None
            else None
        )

        ts = self.config.tile_size
        for y0 in range(0, h, ts):
            y1 = min(y0 + ts, h)
            for x0 in range(0, w, ts):
                x1 = min(x0 + ts, w)
                paste_tile = paste_mask[:, y0:y1, x0:x1]
                tgt_d = (
                    target.depth[:, :, y0:y1, x0:x1]
                    if target.depth is not None
                    else None
                )
                tgt_dv = (
                    target.depth_valid[:, :, y0:y1, x0:x1]
                    if target.depth_valid is not None
                    else None
                )
                src_d = (
                    source.depth[:, :, y0:y1, x0:x1]
                    if source.depth is not None
                    else None
                )
                siv_tile = (
                    src_image_valid[:, y0:y1, x0:x1]
                    if src_image_valid is not None
                    else None
                )
                m_eff = _effective_mask(paste_tile, tgt_d, tgt_dv, src_d, siv_tile)
                m3 = m_eff.unsqueeze(1)

                out_image[:, :, y0:y1, x0:x1] = torch.where(
                    m3, src_imgs[:, :, y0:y1, x0:x1], tgt_imgs[:, :, y0:y1, x0:x1]
                )
                if (
                    out_semantic is not None
                    and src_sem is not None
                    and tgt_sem is not None
                ):
                    out_semantic[:, y0:y1, x0:x1] = torch.where(
                        m_eff, src_sem[:, y0:y1, x0:x1], tgt_sem[:, y0:y1, x0:x1]
                    )
                if (
                    out_panoptic is not None
                    and src_pano is not None
                    and tgt_pano is not None
                ):
                    out_panoptic[:, y0:y1, x0:x1] = torch.where(
                        m_eff, src_pano[:, y0:y1, x0:x1], tgt_pano[:, y0:y1, x0:x1]
                    )
                if out_depth is not None and src_d is not None and tgt_d is not None:
                    out_depth[:, :, y0:y1, x0:x1] = torch.where(m3, src_d, tgt_d)
                if (
                    out_depth_valid is not None
                    and source.depth_valid is not None
                    and tgt_dv is not None
                ):
                    src_v = source.depth_valid[:, :, y0:y1, x0:x1]
                    out_depth_valid[:, :, y0:y1, x0:x1] = torch.where(
                        paste_tile.unsqueeze(1), src_v & tgt_dv, tgt_dv
                    )
                if (
                    out_normals is not None
                    and source.normals is not None
                    and target.normals is not None
                ):
                    out_normals[:, :, y0:y1, x0:x1] = torch.where(
                        m3,
                        source.normals[:, :, y0:y1, x0:x1],
                        target.normals[:, :, y0:y1, x0:x1],
                    )
                if out_target_masks is not None:
                    out_target_masks[:, :, y0:y1, x0:x1] = (
                        out_target_masks[:, :, y0:y1, x0:x1] & ~m3
                    )

        return PaddedBatchedDenseSample(
            images=tv_tensors.Image(out_image),
            boxes=target.boxes,
            labels=target.labels,
            instance_valid=target.instance_valid,
            max_instances=target.max_instances,
            instance_masks=out_target_masks,
            instance_ids=target.instance_ids,
            semantic_maps=(
                SemanticMap(out_semantic) if out_semantic is not None else None
            ),
            panoptic_maps=(
                PanopticMap(out_panoptic) if out_panoptic is not None else None
            ),
            depth=out_depth,
            depth_valid=out_depth_valid,
            normals=out_normals,
            padding_mask=target.padding_mask,
            camera_intrinsics=target.camera_intrinsics,
        )
