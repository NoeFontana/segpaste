"""Type definitions for segpaste package."""

from dataclasses import dataclass
from typing import Literal, Tuple

import torch

from segpaste.compile_util import skip_if_compiling


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


# Type aliases
ImageTensor = torch.Tensor  # [C, H, W]
BoxesTensor = torch.Tensor  # [N, 4]
MasksTensor = torch.Tensor  # [N, H, W]
LabelsTensor = torch.Tensor  # [N]


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


@dataclass(frozen=True, slots=True)
class CopyPasteConfig:
    """Configuration for copy-paste augmentation."""

    # Probability of applying copy-paste augmentation
    paste_probability: float = 0.5

    # Minimum and maximum number of objects to paste from source to target
    min_paste_objects: int = 1
    max_paste_objects: int = 20

    scale_range: Tuple[float, float] = (0.1, 2.0)
    # Blending mode for pasted objects
    blend_mode: Literal["alpha", "gaussian"] = "alpha"

    occluded_area_threshold: float = 0.3  # Objects with >30% occlusion are removed
    box_update_threshold: float = 10.0  # Minimum pixel difference for box updates

    enable_flip: bool = True

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not 0.0 <= self.paste_probability <= 1.0:
            raise ValueError("paste_probability must be between 0.0 and 1.0")
        if self.min_paste_objects > self.max_paste_objects:
            raise ValueError("min_paste_objects must be <= max_paste_objects")
        if self.min_paste_objects < 0:
            raise ValueError("min_paste_objects must be >= 0")
        if self.scale_range[0] > self.scale_range[1]:
            raise ValueError("scale_range min must be <= max")
        if self.blend_mode not in ("alpha", "gaussian", "poisson"):
            raise ValueError("blend_mode must be one of: alpha, gaussian, poisson")
