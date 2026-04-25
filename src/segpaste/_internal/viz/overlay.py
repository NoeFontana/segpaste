"""Per-sample drilldown rendering: orig / aug / overlay tensors.

Each render is a uint8 ``[3, H, W]`` tensor suitable for direct write
via ``torchvision.utils.save_image`` or composition into the contact
sheet by ``torchvision.utils.make_grid``.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torchvision.utils import draw_segmentation_masks

from segpaste.types import DenseSample


def _to_uint8(image: Tensor) -> Tensor:
    return image.clamp(0.0, 1.0).mul(255.0).to(torch.uint8)


def _draw_with_masks(image: Tensor, sample: DenseSample) -> Tensor:
    base = _to_uint8(image.as_subclass(torch.Tensor))
    if sample.instance_masks is None or sample.instance_masks.size(0) == 0:
        return base
    masks = sample.instance_masks.as_subclass(torch.Tensor).to(torch.bool)
    return draw_segmentation_masks(base, masks=masks, alpha=0.5)


def render_drilldown(before: DenseSample, after: DenseSample) -> dict[str, Tensor]:
    """Return the three uint8 renders for one sample's drilldown row."""
    orig = _draw_with_masks(before.image, before)
    aug = _draw_with_masks(after.image, after)

    diff = (
        (after.image.as_subclass(torch.Tensor) - before.image.as_subclass(torch.Tensor))
        .abs()
        .clamp(0.0, 1.0)
    )
    overlay = _to_uint8(diff)

    return {"orig": orig, "aug": aug, "overlay": overlay}
