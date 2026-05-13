"""Per-sample affine ``grid_sample`` propagator (ADR-0008 C4).

Given a ``target`` and a ``source`` :class:`PaddedBatchedDenseSample`
(row-aligned along the batch dim) plus a :class:`BatchedPlacement`,
produces a new :class:`PaddedBatchedDenseSample` where every target ``i``
carries the source sample at ``source_idx[i]`` transformed into the
target's canvas frame under the sampled ``(scale, translate, hflip)``
affine.

The split between target and source is what lets A1 (ADR-0011)
swap intra-batch sources for an external instance bank without changing
the propagator itself: ``target`` provides output canvas geometry and
``source.x[source_idx]`` provides the values to gather. When ``target is
source`` (the default ``IntraBatchSource`` path), behavior is bitwise
identical to the pre-A1 single-input form.

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

from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn.functional import grid_sample
from torchvision import tv_tensors

from segpaste._internal.gpu.batched_placement import BatchedPlacement
from segpaste.compile_util import skip_if_compiling
from segpaste.types import PaddedBatchedDenseSample, PaddingMask
from segpaste.types.dense_sample import PanopticMap, SemanticMap


class WarpedSource(NamedTuple):
    """Audit-only sidecar exposing pre-composite warped source fields.

    Per ADR-0014 §3. Captures the warped source's depth / depth_valid and the
    gathered source ``camera_intrinsics`` before :class:`TileCompositor` merges
    them into the target. Every field is ``None`` when its modality is absent
    on the source. Consumed by :class:`BatchAuditPacket` assembly in
    :meth:`BatchCopyPaste._forward_impl`; not seen by the training hot path.
    """

    warped_depth: Tensor | None
    warped_depth_valid: Tensor | None
    source_intrinsics: Tensor | None


@skip_if_compiling
def _validate_alignment(
    target: PaddedBatchedDenseSample, source: PaddedBatchedDenseSample
) -> None:
    """Assert ``target`` and ``source`` are row-aligned with matching modalities.

    Skipped under ``torch.compile`` (``skip_if_compiling``) so the validation
    cost stays out of the traced graph. The propagator's tensor ops assume
    matching ``B``, ``H``, ``W`` and that any optional modality present on
    ``target`` is also present on ``source`` (so the gather is well-defined).
    """
    if target.batch_size != source.batch_size:
        raise ValueError(
            f"target.batch_size ({target.batch_size}) != "
            f"source.batch_size ({source.batch_size})"
        )
    if target.images.shape[-2:] != source.images.shape[-2:]:
        raise ValueError(
            f"target H,W {tuple(target.images.shape[-2:])} != "
            f"source H,W {tuple(source.images.shape[-2:])}"
        )
    pairs = (
        ("instance_masks", target.instance_masks, source.instance_masks),
        ("instance_ids", target.instance_ids, source.instance_ids),
        ("semantic_maps", target.semantic_maps, source.semantic_maps),
        ("panoptic_maps", target.panoptic_maps, source.panoptic_maps),
        ("depth", target.depth, source.depth),
        ("depth_valid", target.depth_valid, source.depth_valid),
        ("normals", target.normals, source.normals),
        ("padding_mask", target.padding_mask, source.padding_mask),
        ("camera_intrinsics", target.camera_intrinsics, source.camera_intrinsics),
    )
    for name, t, s in pairs:
        if t is not None and s is None:
            raise ValueError(
                f"target carries {name!r} but source does not; "
                "source must declare every modality the target consumes"
            )


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
    src_valid_extent: Tensor,
    device: torch.device,
) -> Tensor:
    """Per-target normalized sampling grid for :func:`F.grid_sample`.

    For an output pixel ``(y, x)``, samples the source at::

        source_y = (y - ty) / scale
        source_x = (x - tx) / scale, then reflected about
                   (W_valid - 1)/2 if ``hflip``

    ``src_valid_extent[i] = (h_v, w_v)`` is the source image's pre-pad extent;
    hflip reflects about ``W_valid - 1`` so right/bottom-pad bands do not shift
    the source content centerline. Grid normalization uses the post-pad ``(h,
    w)`` because ``grid_sample`` reads the post-pad source tensor.

    Returns shape ``[B, H, W, 2]`` normalized for ``align_corners=False``.
    """
    b = translate.size(0)
    ys = torch.arange(h, device=device, dtype=torch.float32)
    xs = torch.arange(w, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    ty = translate[:, 0].view(b, 1, 1)
    tx = translate[:, 1].view(b, 1, 1)
    s = scale.view(b, 1, 1)
    src_w_valid = src_valid_extent[:, 1].view(b, 1, 1)

    src_y = (yy.unsqueeze(0) - ty) / s
    src_x = (xx.unsqueeze(0) - tx) / s
    src_x = torch.where(hflip.view(b, 1, 1), src_w_valid - 1.0 - src_x, src_x)

    grid_x = (src_x + 0.5) * 2.0 / w - 1.0
    grid_y = (src_y + 0.5) * 2.0 / h - 1.0
    return torch.stack([grid_x, grid_y], dim=-1)


def _transform_boxes(
    boxes: Tensor,
    translate: Tensor,
    scale: Tensor,
    hflip: Tensor,
    src_valid_extent: Tensor,
) -> Tensor:
    """Apply the affine to source boxes in xyxy format.

    For hflipped samples x coords reflect about the source's pre-pad
    centerline ``(W_valid - 1)/2`` before the scale+translate; corners may
    swap, so results are canonicalized back to ``x1 <= x2``.
    """
    b, _, _ = boxes.shape
    x1 = boxes[..., 0]
    y1 = boxes[..., 1]
    x2 = boxes[..., 2]
    y2 = boxes[..., 3]

    flip = hflip.view(b, 1)
    src_w_valid = src_valid_extent[:, 1:2]  # [B, 1]
    new_x1 = torch.where(flip, src_w_valid - 1.0 - x2, x1)
    new_x2 = torch.where(flip, src_w_valid - 1.0 - x1, x2)

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

    ``target`` provides the output canvas geometry (``B``, ``K``, ``H``, ``W``,
    ``device``); every modality value is gathered from ``source.x[source_idx]``.
    Pass ``target`` and ``source`` as the same object to recover the pre-A1
    intra-batch behavior bitwise.
    """

    def forward(
        self,
        target: PaddedBatchedDenseSample,
        source: PaddedBatchedDenseSample,
        placement: BatchedPlacement,
    ) -> tuple[PaddedBatchedDenseSample, WarpedSource]:
        _validate_alignment(target, source)
        b = target.batch_size
        # ``k`` is the source view's K — the output carries one warped row
        # per source slot. For ``IntraBatchSource``, ``source is target`` so
        # ``k == target.max_instances`` and behavior matches v0.3.0.
        k = source.max_instances
        device = target.images.device
        _, _, h, w = target.images.shape

        if b == 0:
            return target, WarpedSource(None, None, None)

        grid = _build_grid(
            h,
            w,
            placement.translate,
            placement.scale,
            placement.hflip,
            placement.src_valid_extent,
            device,
        )

        src_images = source.images[placement.source_idx]
        warped_images = grid_sample(
            src_images, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )

        warped_boxes = _transform_boxes(
            source.boxes[placement.source_idx],
            placement.translate,
            placement.scale,
            placement.hflip,
            placement.src_valid_extent,
        )
        warped_labels = source.labels[placement.source_idx]

        warped_masks: Tensor | None = None
        warped_ids: Tensor | None = None
        if source.instance_masks is not None and source.instance_ids is not None:
            src_masks = source.instance_masks[placement.source_idx].float()
            sampled = grid_sample(
                src_masks,
                grid,
                mode="nearest",
                padding_mode="zeros",
                align_corners=False,
            )
            warped_masks = sampled > 0.5
            warped_ids = source.instance_ids[placement.source_idx]

        warped_semantic: SemanticMap | None = None
        if source.semantic_maps is not None:
            sem = source.semantic_maps[placement.source_idx]
            sampled = grid_sample(
                sem.unsqueeze(1).float(),
                grid,
                mode="nearest",
                padding_mode="zeros",
                align_corners=False,
            )
            warped_semantic = SemanticMap(sampled.squeeze(1).long())

        warped_panoptic: PanopticMap | None = None
        if source.panoptic_maps is not None:
            pano = source.panoptic_maps[placement.source_idx]
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
        if source.depth is not None and source.depth_valid is not None:
            src_depth = source.depth[placement.source_idx]
            warped_depth = grid_sample(
                src_depth,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            warped_depth_valid = _warp_validity_gate(
                source.depth_valid[placement.source_idx], grid
            )

        warped_normals: Tensor | None = None
        if source.normals is not None:
            src_normals = source.normals[placement.source_idx]
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
            source.camera_intrinsics[placement.source_idx]
            if source.camera_intrinsics is not None
            else None
        )

        warped_padding_mask: PaddingMask | None = None
        if source.padding_mask is not None:
            src_iv = ~source.padding_mask[placement.source_idx].as_subclass(Tensor)
            warped_iv = _warp_validity_gate(src_iv, grid)
            warped_padding_mask = PaddingMask.from_tensor(~warped_iv)

        propagated = PaddedBatchedDenseSample(
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
        audit = WarpedSource(
            warped_depth=warped_depth,
            warped_depth_valid=warped_depth_valid,
            source_intrinsics=warped_intrinsics,
        )
        return propagated, audit
