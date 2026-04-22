"""Data structures for detection targets and bounding boxes."""

from dataclasses import dataclass
from typing import Any

from torchvision.tv_tensors import Mask

from segpaste.compile_util import skip_if_compiling
from segpaste.types.type_aliases import (
    BoxesTensor,
    ImageTensor,
    LabelsTensor,
    MasksTensor,
)


class PaddingMask(Mask):
    """Unlike tv_tensor.Mask, PaddingMask is not associated with any object.

    It is used to indicate padded parts of an Image. Unlike tv_tensor.Mask, it is
    is forwarded unchanged by this package reimplementation SanitizeBoundingBoxes.
    """

    pass


@dataclass(slots=True)  # Not frozen since it may be modified during augmentation
class DetectionTarget:
    """Detection target containing image and annotations."""

    image: ImageTensor  # Shape: [C, H, W]
    boxes: BoxesTensor  # Shape: [N, 4], format: xyxy (top, left, bottom, right)
    labels: LabelsTensor  # Shape: [N]
    masks: MasksTensor  # Shape: [N, H, W]
    padding_mask: PaddingMask | None = None  # Shape: [1, H, W]

    TYPES = ImageTensor | BoxesTensor | LabelsTensor | MasksTensor | PaddingMask | None

    @skip_if_compiling
    def __post_init__(self) -> None:
        """Validate tensor shapes and consistency."""
        if self.boxes.size(0) != self.labels.size(0):
            raise ValueError("boxes and labels must have same number of objects")
        if self.masks.size(0) != self.boxes.size(0):
            raise ValueError("masks and boxes must have same number of objects")
        if (
            self.padding_mask is not None
            and self.padding_mask.shape[1:] != self.image.shape[1:]
        ):
            raise ValueError("padding_mask must have same height and width as image")

    def to_dict(
        self,
    ) -> dict[str, TYPES]:
        """Convert DetectionTarget to a dictionary."""
        return {
            "image": self.image,
            "boxes": self.boxes,
            "labels": self.labels,
            "masks": self.masks,
            "padding_mask": self.padding_mask,
        }

    @staticmethod
    def from_dict(
        data: dict[str, Any],
    ) -> "DetectionTarget":
        """Create DetectionTarget from a dictionary."""
        return DetectionTarget(
            image=data["image"],
            boxes=data["boxes"],
            labels=data["labels"],
            masks=data["masks"],
            padding_mask=data.get("padding_mask"),
        )
