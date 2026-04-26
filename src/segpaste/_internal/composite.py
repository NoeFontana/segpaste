"""The :class:`DenseComposite` primitive — ADR-0005 §1.

Depth-buffered per-sample where-composite. Retained as the reference
pixelwise-where kernel; the batched GPU lane in
:mod:`segpaste.augmentation.batch_copy_paste` inlines the same z-test
and where-fusion semantics for speed.

Kept under ``segpaste._internal`` for W2; see ADR-0005 §5 for promotion
policy.
"""

from __future__ import annotations

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor, nn
from torchvision import tv_tensors
from torchvision.ops import masks_to_boxes

from segpaste.types import DenseSample, InstanceMask, PanopticMap, SemanticMap


class CompositeConfig(BaseModel):
    """Configuration for :class:`DenseComposite`."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    min_composited_area: int = Field(default=50, ge=0)
    """Drop pasted-or-survivor instance masks whose area is below this.

    Enforces ADR-0001 Part (ii)'s small-area invariant. Survivor masks are
    measured against their original area ratio; pasted masks are measured
    by absolute pixel count.
    """

    occluded_area_threshold: float = Field(default=0.99, ge=0.0, le=1.0)
    """Drop survivor objects whose occlusion ratio exceeds this threshold.

    ``occlusion_ratio = 1 - updated_area / original_area``. A value of 1.0
    disables the filter.
    """


class DenseComposite(nn.Module):
    """Depth-buffered where-composite primitive.

    ``forward(target, source, paste_mask)`` returns a composed
    :class:`DenseSample` in which the active modalities on either input
    are composited pixelwise under the effective paste mask
    ``M_eff = paste_mask & (z-test)``. When depth is absent on either
    input the z-test degenerates and ``M_eff = paste_mask``, recovering
    the Ghiasi alpha composite exactly (see ADR-0005 §4 parity gate).
    """

    config: CompositeConfig

    def __init__(self, config: CompositeConfig) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        target: DenseSample,
        source: DenseSample,
        paste_mask: Tensor,
    ) -> DenseSample:
        if paste_mask.dtype != torch.bool:
            raise ValueError("paste_mask must be bool")
        if paste_mask.shape != target.image.shape[-2:]:
            raise ValueError("paste_mask must share H, W with target.image")

        m_eff = self._effective_mask(target, source, paste_mask)
        image_out = self._composite_image(target, source, m_eff)

        (
            inst_masks_out,
            inst_ids_out,
            labels_out,
            boxes_out,
        ) = self._composite_instances(target, source, m_eff)

        semantic_out = self._composite_semantic(target, source, m_eff)
        panoptic_out = self._composite_panoptic(target, source, m_eff)
        depth_out, depth_valid_out = self._composite_depth(
            target, source, m_eff, paste_mask
        )
        normals_out = self._composite_normals(target, source, m_eff)

        h, w = image_out.shape[-2:]
        return DenseSample(
            image=tv_tensors.Image(image_out),
            boxes=tv_tensors.BoundingBoxes(  # pyright: ignore[reportCallIssue]
                boxes_out,
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=(h, w),
            ),
            labels=labels_out,
            instance_ids=inst_ids_out,
            instance_masks=(
                InstanceMask(inst_masks_out) if inst_masks_out is not None else None
            ),
            semantic_map=semantic_out,
            panoptic_map=panoptic_out,
            depth=depth_out,
            depth_valid=depth_valid_out,
            normals=normals_out,
            padding_mask=target.padding_mask,
            camera_intrinsics=target.camera_intrinsics,
            metric_depth=target.metric_depth,
        )

    def _effective_mask(
        self, target: DenseSample, source: DenseSample, paste_mask: Tensor
    ) -> Tensor:
        if target.depth is None or source.depth is None:
            m_eff = paste_mask
        else:
            # depth/depth_valid co-optional invariant is enforced by DenseSample.
            tgt_valid = target.depth_valid
            if tgt_valid is None:  # pragma: no cover - guarded by DenseSample
                raise ValueError("target.depth_valid missing alongside target.depth")
            closer = (source.depth < target.depth).squeeze(0)
            invalid = (~tgt_valid).squeeze(0)
            m_eff = paste_mask & (closer | invalid)
        if source.padding_mask is not None:
            m_eff = m_eff & ~source.padding_mask.as_subclass(Tensor).squeeze(0)
        return m_eff

    def _composite_image(
        self, target: DenseSample, source: DenseSample, m_eff: Tensor
    ) -> Tensor:
        tgt_img = target.image.as_subclass(Tensor)
        src_img = source.image.as_subclass(Tensor)
        return torch.where(m_eff.unsqueeze(0), src_img, tgt_img)

    def _composite_instances(
        self, target: DenseSample, source: DenseSample, m_eff: Tensor
    ) -> tuple[Tensor | None, Tensor | None, Tensor, Tensor]:
        """Stack survivor + pasted instance rows.

        Returns ``(masks, ids, labels, boxes)`` or ``(None, None, labels, boxes)``
        when neither sample carries INSTANCE.
        """
        tgt_has_inst = target.instance_masks is not None
        src_has_inst = source.instance_masks is not None
        if not tgt_has_inst and not src_has_inst:
            return None, None, target.labels, target.boxes.as_subclass(Tensor)

        pasted_masks = (
            source.instance_masks.as_subclass(Tensor).to(torch.bool)
            if source.instance_masks is not None
            else torch.empty(
                (0, *target.image.shape[-2:]),
                dtype=torch.bool,
                device=target.image.device,
            )
        )
        pasted_labels = (
            source.labels
            if src_has_inst
            else torch.empty(
                (0,), dtype=target.labels.dtype, device=target.labels.device
            )
        )

        if tgt_has_inst:
            if target.instance_masks is None or target.instance_ids is None:
                raise ValueError("target.instance_masks/ids co-optional invariant")
            original_masks = target.instance_masks.as_subclass(Tensor).to(torch.bool)
            updated_masks = original_masks & ~m_eff
            valid_indices = self._survivor_indices(original_masks, updated_masks)
            survivor_masks = updated_masks[valid_indices]
            survivor_labels = target.labels[valid_indices]
            survivor_ids = target.instance_ids[valid_indices]
            max_prev = (
                int(target.instance_ids.max().item())
                if target.instance_ids.numel() > 0
                else -1
            )
        else:
            h, w = target.image.shape[-2:]
            survivor_masks = torch.empty(
                (0, h, w), dtype=torch.bool, device=target.image.device
            )
            survivor_labels = torch.empty(
                (0,), dtype=target.labels.dtype, device=target.labels.device
            )
            survivor_ids = torch.empty(
                (0,), dtype=torch.int32, device=target.image.device
            )
            max_prev = -1

        k = int(pasted_masks.shape[0])
        new_ids = torch.arange(
            max_prev + 1,
            max_prev + 1 + k,
            dtype=torch.int32,
            device=survivor_ids.device,
        )

        all_masks = torch.cat([survivor_masks, pasted_masks], dim=0)
        all_labels = torch.cat([survivor_labels, pasted_labels], dim=0)
        all_ids = torch.cat([survivor_ids, new_ids], dim=0)

        if all_masks.shape[0] == 0:
            all_boxes = torch.empty(
                (0, 4), dtype=target.boxes.dtype, device=target.boxes.device
            )
        else:
            all_boxes = masks_to_boxes(all_masks).to(target.boxes.dtype)

        return all_masks, all_ids, all_labels, all_boxes

    def _survivor_indices(
        self, original_masks: Tensor, updated_masks: Tensor
    ) -> Tensor:
        if original_masks.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=original_masks.device)
        original_areas = original_masks.sum(dim=(1, 2))
        updated_areas = updated_masks.sum(dim=(1, 2))
        has_area = updated_areas > 0
        min_area_ok = updated_areas >= self.config.min_composited_area
        occlusion_ratio = 1.0 - (
            updated_areas.to(torch.float32) / (original_areas.to(torch.float32) + 1e-8)
        )
        not_too_occluded = occlusion_ratio <= self.config.occluded_area_threshold
        return torch.where(has_area & min_area_ok & not_too_occluded)[0]

    def _composite_semantic(
        self, target: DenseSample, source: DenseSample, m_eff: Tensor
    ) -> SemanticMap | None:
        if target.semantic_map is None and source.semantic_map is None:
            return None
        if target.semantic_map is None or source.semantic_map is None:
            raise ValueError(
                "semantic_map must be present on both target and source, or neither"
            )
        tgt = target.semantic_map.as_subclass(Tensor)
        src = source.semantic_map.as_subclass(Tensor)
        out = torch.where(m_eff, src, tgt)
        return SemanticMap(out)

    def _composite_panoptic(
        self, target: DenseSample, source: DenseSample, m_eff: Tensor
    ) -> PanopticMap | None:
        if target.panoptic_map is None and source.panoptic_map is None:
            return None
        if target.panoptic_map is None or source.panoptic_map is None:
            raise ValueError(
                "panoptic_map must be present on both target and source, or neither"
            )
        tgt = target.panoptic_map.as_subclass(Tensor)
        src = source.panoptic_map.as_subclass(Tensor)
        out = torch.where(m_eff, src, tgt)
        return PanopticMap(out)

    def _composite_depth(
        self,
        target: DenseSample,
        source: DenseSample,
        m_eff: Tensor,
        paste_mask: Tensor,
    ) -> tuple[Tensor | None, Tensor | None]:
        """Emit ``(depth, depth_valid)`` per ADR-0007 §3 and ADR-0001 §(ii).

        ``d_out = where(m_eff, d_src, d_tgt)`` — since ``m_eff`` already
        encodes the z-test, this is bitwise-identical to
        ``min(d_src, d_tgt)`` on the paste footprint. Validity is
        piecewise: ``V_tgt`` outside ``paste_mask``, ``V_src ∧ V_tgt``
        inside (ADR-0001 §(ii) amended by ADR-0007 §2).
        """
        if target.depth is None and source.depth is None:
            return None, None
        if target.depth is None or source.depth is None:
            raise ValueError(
                "depth must be present on both target and source, or neither"
            )
        if target.depth_valid is None or source.depth_valid is None:
            raise ValueError(
                "depth_valid must accompany depth on both target and source"
            )
        m3 = m_eff.unsqueeze(0)
        p3 = paste_mask.unsqueeze(0)
        depth_out = torch.where(m3, source.depth, target.depth)
        valid_out = torch.where(
            p3, source.depth_valid & target.depth_valid, target.depth_valid
        )
        return depth_out, valid_out

    def _composite_normals(
        self, target: DenseSample, source: DenseSample, m_eff: Tensor
    ) -> Tensor | None:
        if target.normals is None and source.normals is None:
            return None
        if target.normals is None or source.normals is None:
            raise ValueError(
                "normals must be present on both target and source, or neither"
            )
        return torch.where(m_eff.unsqueeze(0), source.normals, target.normals)
