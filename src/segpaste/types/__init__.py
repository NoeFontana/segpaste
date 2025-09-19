"""Type definitions for the segpaste package.

This module contains all data structures and configuration types used
throughout the package.
"""

from segpaste.types.data_structures import BoundingBox, DetectionTarget
from segpaste.types.type_aliases import (
    BoxesTensor,
    ImageTensor,
    LabelsTensor,
    MasksTensor,
)

__all__ = [
    "BoundingBox",
    "DetectionTarget",
    "ImageTensor",
    "BoxesTensor",
    "MasksTensor",
    "LabelsTensor",
]
