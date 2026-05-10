"""Right/bottom canvas pad to a multiple of ``p`` (A2).

Brings the spatial extent of a :class:`PaddedBatchedDenseSample` up to the
smallest multiple of ``p`` covering ``(H, W)``. Reflect-pads images so the
ViT patch-embed does not see a hard zero edge masquerading as content; all
other modalities take constant pads (``False`` for bool, ``0`` for depth/
normals, ``ignore_index`` for semantic/panoptic, ``True`` for the padding
mask). ``boxes``, ``instance_valid``, and ``camera_intrinsics`` do not
depend on canvas extent and are forwarded unchanged.

Pad amounts ``(pad_h, pad_w) = (-H % p, -W % p)`` are Python ints so the
trace specializes on the post-pad canvas size — fine for static-shape
training where (H, W) come from a fixed LSJ output. Already-divisible
inputs return the input object unchanged.
"""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision import tv_tensors

from segpaste.types import PaddedBatchedDenseSample, PaddingMask
from segpaste.types.dense_sample import PanopticMap, SemanticMap


def pad_canvas_to_multiple(
    padded: PaddedBatchedDenseSample,
    p: int,
    ignore_index: int,
) -> PaddedBatchedDenseSample:
    """Pad ``(H, W)`` up to multiples of ``p`` on the right and bottom edges.

    ``ignore_index`` fills the new band of ``semantic_maps`` and
    ``panoptic_maps``. The fill follows the LSJ convention (255 by default).
    """
    if p <= 0:
        raise ValueError(f"p must be positive, got {p}")

    b, _, h, w = padded.images.shape
    pad_h = (-h) % p
    pad_w = (-w) % p
    if pad_h == 0 and pad_w == 0:
        return padded

    pads = (0, pad_w, 0, pad_h)  # (left, right, top, bottom) for last 2 dims

    def const(t: Tensor | None, value: float | bool) -> Tensor | None:
        return None if t is None else F.pad(t, pads, mode="constant", value=value)

    images_t = padded.images.as_subclass(Tensor)
    # ``mode="reflect"`` requires each pad amount strictly less than the
    # corresponding axis size. Guard the pathological tiny-canvas case.
    if pad_h < h and pad_w < w:
        new_images_t = F.pad(images_t, pads, mode="reflect")
    else:
        new_images_t = F.pad(images_t, pads, mode="constant", value=0.0)

    sem = padded.semantic_maps
    pano = padded.panoptic_maps
    base_pad_mask = (
        padded.padding_mask.as_subclass(Tensor)
        if padded.padding_mask is not None
        else torch.zeros((b, 1, h, w), dtype=torch.bool, device=padded.images.device)
    )

    return replace(
        padded,
        images=tv_tensors.Image(new_images_t),
        instance_masks=const(padded.instance_masks, False),
        semantic_maps=(
            None
            if sem is None
            else SemanticMap(
                F.pad(
                    sem.as_subclass(Tensor), pads, mode="constant", value=ignore_index
                )
            )
        ),
        panoptic_maps=(
            None
            if pano is None
            else PanopticMap(
                F.pad(
                    pano.as_subclass(Tensor), pads, mode="constant", value=ignore_index
                )
            )
        ),
        depth=const(padded.depth, 0.0),
        depth_valid=const(padded.depth_valid, False),
        normals=const(padded.normals, 0.0),
        padding_mask=PaddingMask.from_tensor(
            F.pad(base_pad_mask, pads, mode="constant", value=True)
        ),
    )
