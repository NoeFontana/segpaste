"""Image-level data structures (padding mask)."""

from torchvision.tv_tensors import Mask


class PaddingMask(Mask):
    """Unlike tv_tensor.Mask, PaddingMask is not associated with any object.

    It is used to indicate padded parts of an Image. Unlike tv_tensor.Mask, it is
    is forwarded unchanged by this package reimplementation SanitizeBoundingBoxes.
    """

    pass
