"""Configuration classes for copy-paste augmentation."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CopyPasteConfig(BaseModel):
    """Configuration for copy-paste augmentation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Probability of applying copy-paste augmentation
    paste_probability: float = Field(0.5, ge=0.0, le=1.0)

    # Minimum and maximum number of objects to paste from source to target
    min_paste_objects: int = Field(1, ge=0)
    max_paste_objects: int = Field(5, ge=0)

    scale_range: tuple[float, float] = (0.5, 2.0)
    # Blending mode for pasted objects
    blend_mode: Literal["alpha", "gaussian", "poisson"] = "alpha"

    # Min edge length of pasted objects after scaling
    # Bounding boxes with smaller edges will be skipped
    min_object_edge: int = 10

    # Min mask area of pasted objects after scaling
    # Masks with smaller area will be skipped
    min_object_area: int = 50

    # After pasting, objects with an occlusion ratio above this threshold are removed
    # Occluded area ratio = 1 - (visible area / original area)
    # Set to 1.0 to disable this filtering
    occluded_area_threshold: float = 0.99

    @model_validator(mode="after")
    def validate_config(self) -> "CopyPasteConfig":
        """Validate configuration parameters."""
        if self.min_paste_objects > self.max_paste_objects:
            raise ValueError("min_paste_objects must be <= max_paste_objects")

        if self.scale_range[0] > self.scale_range[1]:
            raise ValueError("scale_range min must be <= max")

        return self
