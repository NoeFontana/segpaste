"""Data structures for detection targets and bounding boxes."""

from dataclasses import dataclass

from torchvision.tv_tensors import Mask

from segpaste.compile_util import skip_if_compiling
from segpaste.types.type_aliases import (
    BoxesTensor,
    ImageTensor,
    LabelsTensor,
    MasksTensor,
)


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Bounding box in xyxy format (x1, y1, x2, y2)."""

    x1: float
    y1: float
    x2: float
    y2: float

    @skip_if_compiling
    def __post_init__(self) -> None:
        """Validate bounding box coordinates."""
        if self.x1 >= self.x2:
            raise ValueError("x1 must be < x2")
        if self.y1 >= self.y2:
            raise ValueError("y1 must be < y2")


class PaddingMask(Mask):  # type: ignore[misc]
    """Unlike tv_tensor.Mask, PaddingMask is not associated with any object.

    It is used to indicate padded parts of an Image. Unlike tv_tensor.Mask, it is
    is forwarded unchanged by this package reimplementation SanitizeBoundingBoxes.
    """

    pass


@dataclass(slots=True)  # Not frozen since it may be modified during augmentation
class DetectionTarget:
    """Detection target containing image and annotations."""

    image: ImageTensor  # Shape: [C, H, W]
    boxes: BoxesTensor  # Shape: [N, 4], format: xyxy
    labels: LabelsTensor  # Shape: [N]
    masks: MasksTensor  # Shape: [N, H, W]

    @skip_if_compiling
    def __post_init__(self) -> None:
        """Validate tensor shapes and consistency."""
        if self.boxes.size(0) != self.labels.size(0):
            raise ValueError("boxes and labels must have same number of objects")
        if self.masks.size(0) != self.boxes.size(0):
            raise ValueError("masks and boxes must have same number of objects")
