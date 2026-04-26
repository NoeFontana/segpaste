"""Image-level data structures (padding mask)."""

from typing import cast

import torch
from torchvision.tv_tensors import Mask


class PaddingMask(Mask):
    """Unlike tv_tensor.Mask, PaddingMask is not associated with any object.

    It is used to indicate padded parts of an Image. Unlike tv_tensor.Mask, it is
    is forwarded unchanged by this package reimplementation SanitizeBoundingBoxes.
    """

    @classmethod
    def from_tensor(cls, data: torch.Tensor) -> "PaddingMask":
        """Wrap a bool tensor as :class:`PaddingMask` with the static type preserved.

        ``Mask.__new__`` is annotated to return ``Mask``; this factory exists
        purely to recover ``PaddingMask`` typing without smearing ``cast`` at
        every call site.
        """
        return cast(PaddingMask, cls(data))
