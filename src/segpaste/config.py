"""Configuration classes for copy-paste augmentation."""

from dataclasses import dataclass
from typing import Literal, Tuple


@dataclass(frozen=True, slots=True)
class CopyPasteConfig:
    """Configuration for copy-paste augmentation."""

    # Probability of applying copy-paste augmentation
    paste_probability: float = 0.5

    # Minimum and maximum number of objects to paste from source to target
    min_paste_objects: int = 1
    max_paste_objects: int = 5

    scale_range: Tuple[float, float] = (0.5, 2.0)
    # Blending mode for pasted objects
    blend_mode: Literal["alpha", "gaussian"] = "alpha"

    occluded_area_threshold: float = 0.99  # Objects with >99% occlusion are removed

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
