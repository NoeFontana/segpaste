"""Type definitions for segpaste package."""

from dataclasses import dataclass
from typing import Optional, Tuple

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
    masks: Optional[MasksTensor] = None  # Shape: [N, H, W]

    @skip_if_compiling
    def __post_init__(self) -> None:
        """Validate tensor shapes and consistency."""
        if self.image.dim() != 3:
            raise ValueError("image must be 3D tensor [C, H, W]")
        if self.boxes.dim() != 2 or self.boxes.size(1) != 4:
            raise ValueError("boxes must be 2D tensor [N, 4]")
        if self.labels.dim() != 1:
            raise ValueError("labels must be 1D tensor [N]")
        if self.boxes.size(0) != self.labels.size(0):
            raise ValueError("boxes and labels must have same number of objects")
        if self.masks is not None:
            if self.masks.dim() != 3:
                raise ValueError("masks must be 3D tensor [N, H, W]")
            if self.masks.size(0) != self.boxes.size(0):
                raise ValueError("masks and boxes must have same number of objects")


@dataclass(frozen=True, slots=True)
class CopyPasteConfig:
    """Configuration for copy-paste augmentation."""

    paste_probability: float = 0.5
    max_paste_objects: int = 10
    min_paste_objects: int = 1
    scale_range: Tuple[float, float] = (0.1, 2.0)
    blend_mode: str = "alpha"
    occluded_area_threshold: float = 0.3  # Objects with >30% occlusion are removed
    box_update_threshold: float = 10.0  # Minimum pixel difference for box updates
    enable_rotation: bool = False
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
