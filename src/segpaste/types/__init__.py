"""Type definitions for the segpaste package.

This module contains all data structures and configuration types used
throughout the package.
"""

from segpaste.types.data_structures import BoundingBox, DetectionTarget, PaddingMask
from segpaste.types.dense_sample import (
    CameraIntrinsics,
    DenseSample,
    InstanceMask,
    Modality,
    PanopticMap,
    PanopticSchema,
    SemanticMap,
)
from segpaste.types.type_aliases import (
    BoxesTensor,
    ImageTensor,
    LabelsTensor,
    MasksTensor,
)

__all__ = [
    "BoundingBox",
    "BoxesTensor",
    "CameraIntrinsics",
    "DenseSample",
    "DetectionTarget",
    "ImageTensor",
    "InstanceMask",
    "LabelsTensor",
    "MasksTensor",
    "Modality",
    "PaddingMask",
    "PanopticMap",
    "PanopticSchema",
    "SemanticMap",
]
