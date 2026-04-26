"""Per-sample affine ``grid_sample`` propagator (ADR-0008 C4).

Given a :class:`PaddedBatchedDenseSample` and a :class:`BatchedPlacement`,
produces a new :class:`PaddedBatchedDenseSample` where every target ``i``
carries the source sample at ``source_idx[i]`` transformed into the target
frame under the sampled ``(scale, translate, hflip)`` affine.

Channel-group sampling modes (ADR-0008 §5):

* image / depth / normals — bilinear
* instance_masks / semantic_maps / panoptic_maps / depth_valid — nearest

Corrections:

* normals — bilinear spatial sampling plus x-sign negation on ``hflip``
  rows (ADR-0007 §7). Ray-rectification under translation is deferred.
* depth_valid — zero-padded outside the transformed footprint; out-of-frame
  regions inherit ``False`` so depth-aware composites treat them as missing.

``align_corners=False`` fixes the grid convention at pixel centers so
nearest-mode sampling does not flip bool boundaries on sub-pixel offsets.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn.functional import grid_sample
from torchvision import tv_tensors

from segpaste._internal.gpu.batched_placement import BatchedPlacement
from segpaste.types import PaddedBatchedDenseSample, PaddingMask
from segpaste.types.dense_sample import PanopticMap, SemanticMap


def _warp_validity_gate(src_gate: Tensor, grid: Tensor) -> Tensor:
    """Warp a bool validity gate (``True`` = valid) via nearest-neighbor sampling.

    Pixels mapped from outside the source frame come back as 0 under
    ``padding_mode="zeros"`` — i.e. invalid in the output, which matches
    the desired semantics for both ``depth_valid`` and ``image_valid``.
    """
    sampled = grid_sample(
        src_gate.to(torch.float32),
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=False,
    )
    return sampled > 0.5


def _build_grid(
    h: int,
    w: int,
    translate: Tensor,
    scale: Tensor,
    hflip: Tensor,
    device: torch.device,
) -> Tensor:
    """Per-target normalized sampling grid for :func:`F.grid_sample`.

    For an output pixel ``(y, x)``, samples the source at::

        source_y = (y - ty) / scale
        source_x = (x - tx) / scale, then reflected if ``hflip``

    Returns shape ``[B, H, W, 2]`` normalized for ``align_corners=False``.
    """
    b = translate.size(0)
    ys = torch.arange(h, device=device, dtype=torch.float32)
    xs = torch.arange(w, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    ty = translate[:, 0].view(b, 1, 1)
    tx = translate[:, 1].view(b, 1, 1)
    s = scale.view(b, 1, 1)

    src_y = (yy.unsqueeze(0) - ty) / s
    src_x = (xx.unsqueeze(0) - tx) / s
    src_x = torch.where(hflip.view(b, 1, 1), (w - 1.0) - src_x, src_x)

    grid_x = (src_x + 0.5) * 2.0 / w - 1.0
    grid_y = (src_y + 0.5) * 2.0 / h - 1.0
    return torch.stack([grid_x, grid_y], dim=-1)


def _transform_boxes(
    boxes: Tensor,
    translate: Tensor,
    scale: Tensor,
    hflip: Tensor,
    w: int,
) -> Tensor:
    """Apply the affine to source boxes in xyxy format.

    For hflipped samples x coords reflect about the source-image centerline
    (``x' = (W - 1) - x``) before the scale+translate; corners may swap, so
    results are canonicalized back to ``x1 <= x2``.
    """
    b, _, _ = boxes.shape
    x1 = boxes[..., 0]
    y1 = boxes[..., 1]
    x2 = boxes[..., 2]
    y2 = boxes[..., 3]

    flip = hflip.view(b, 1)
    new_x1 = torch.where(flip, (w - 1.0) - x2, x1)
    new_x2 = torch.where(flip, (w - 1.0) - x1, x2)

    s = scale.view(b, 1)
    ty = translate[:, 0:1]
    tx = translate[:, 1:2]

    out_x1 = new_x1 * s + tx
    out_x2 = new_x2 * s + tx
    out_y1 = y1 * s + ty
    out_y2 = y2 * s + ty

    return torch.stack([out_x1, out_y1, out_x2, out_y2], dim=-1)


class AffinePropagator(nn.Module):
    """Per-sample affine propagator consumed by :class:`BatchCopyPaste`.

    Output ``PaddedBatchedDenseSample`` holds the transformed source at each
    target index; its ``instance_valid`` mirrors ``placement.paste_valid``.
    """

    def forward(
        self,
        padded: PaddedBatchedDenseSample,
        placement: BatchedPlacement,
    ) -> PaddedBatchedDenseSample:
        b = padded.batch_size
        k = padded.max_instances
        device = padded.images.device
        _, _, h, w = padded.images.shape

        if b == 0:
            return padded

        grid = _build_grid(
            h, w, placement.translate, placement.scale, placement.hflip, device
        )

        src_images = padded.images[placement.source_idx]
        warped_images = grid_sample(
            src_images, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )

        warped_boxes = _transform_boxes(
            padded.boxes[placement.source_idx],
            placement.translate,
            placement.scale,
            placement.hflip,
            w,
        )
        warped_labels = padded.labels[placement.source_idx]

        warped_masks: Tensor | None = None
        warped_ids: Tensor | None = None
        if padded.instance_masks is not None and padded.instance_ids is not None:
            src_masks = padded.instance_masks[placement.source_idx].float()
            sampled = grid_sample(
                src_masks,
                grid,
                mode="nearest",
                padding_mode="zeros",
                align_corners=False,
            )
            warped_masks = sampled > 0.5
            warped_ids = padded.instance_ids[placement.source_idx]

        warped_semantic: SemanticMap | None = None
        if padded.semantic_maps is not None:
            sem = padded.semantic_maps[placement.source_idx]
            sampled = grid_sample(
                sem.unsqueeze(1).float(),
                grid,
                mode="nearest",
                padding_mode="zeros",
                align_corners=False,
            )
            warped_semantic = SemanticMap(sampled.squeeze(1).long())

        warped_panoptic: PanopticMap | None = None
        if padded.panoptic_maps is not None:
            pano = padded.panoptic_maps[placement.source_idx]
            sampled = grid_sample(
                pano.unsqueeze(1).float(),
                grid,
                mode="nearest",
                padding_mode="zeros",
                align_corners=False,
            )
            warped_panoptic = PanopticMap(sampled.squeeze(1).long())

        warped_depth: Tensor | None = None
        warped_depth_valid: Tensor | None = None
        if padded.depth is not None and padded.depth_valid is not None:
            src_depth = padded.depth[placement.source_idx]
            warped_depth = grid_sample(
                src_depth,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            warped_depth_valid = _warp_validity_gate(
                padded.depth_valid[placement.source_idx], grid
            )

        warped_normals: Tensor | None = None
        if padded.normals is not None:
            src_normals = padded.normals[placement.source_idx]
            warped_n = grid_sample(
                src_normals,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            flip_sign = torch.where(placement.hflip.view(b, 1, 1, 1), -1.0, 1.0)
            ones = torch.ones_like(flip_sign)
            sign = torch.cat([flip_sign, ones, ones], dim=1)
            warped_normals = warped_n * sign

        warped_intrinsics: Tensor | None = (
            padded.camera_intrinsics[placement.source_idx]
            if padded.camera_intrinsics is not None
            else None
        )

        warped_padding_mask: PaddingMask | None = None
        if padded.padding_mask is not None:
            src_iv = ~padded.padding_mask[placement.source_idx].as_subclass(Tensor)
            warped_iv = _warp_validity_gate(src_iv, grid)
            warped_padding_mask = PaddingMask.from_tensor(~warped_iv)

        return PaddedBatchedDenseSample(
            images=tv_tensors.Image(warped_images),
            boxes=warped_boxes,
            labels=warped_labels,
            instance_valid=placement.paste_valid,
            max_instances=k,
            instance_masks=warped_masks,
            instance_ids=warped_ids,
            semantic_maps=warped_semantic,
            panoptic_maps=warped_panoptic,
            depth=warped_depth,
            depth_valid=warped_depth_valid,
            normals=warped_normals,
            padding_mask=warped_padding_mask,
            camera_intrinsics=warped_intrinsics,
        )
