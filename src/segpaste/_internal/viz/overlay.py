"""Per-sample drilldown rendering: orig / aug / overlay tensors.

Each render is a uint8 ``[3, H, W]`` tensor suitable for direct write
via ``torchvision.utils.save_image`` or composition into the contact
sheet by ``torchvision.utils.make_grid``. ``orig`` / ``aug`` are the
raw image pixels — instance masks are exposed through FiftyOne's
native ``Detections`` field at export time, not baked into pixels.
"""

from __future__ import annotations

import torch
from torch import Tensor

from segpaste.types import DenseSample


def _to_uint8(image: Tensor) -> Tensor:
    return image.clamp(0.0, 1.0).mul(255.0).to(torch.uint8)


def render_drilldown(before: DenseSample, after: DenseSample) -> dict[str, Tensor]:
    """Return the three uint8 renders for one sample's drilldown row."""
    orig = _to_uint8(before.image.as_subclass(torch.Tensor))
    aug = _to_uint8(after.image.as_subclass(torch.Tensor))

    diff = (
        (after.image.as_subclass(torch.Tensor) - before.image.as_subclass(torch.Tensor))
        .abs()
        .clamp(0.0, 1.0)
    )
    overlay = _to_uint8(diff)

    return {"orig": orig, "aug": aug, "overlay": overlay}
