"""Type definitions for the segpaste package.

This module contains all data structures and configuration types used
throughout the package.
"""

# DetectionTarget is internal-only as of 0.9.0 (ADR-0003); `as X` marks the
# import as an intentional re-export kept off __all__ for its internal callers.
from segpaste.types.data_structures import DetectionTarget as DetectionTarget
from segpaste.types.data_structures import PaddingMask
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
    "BoxesTensor",
    "CameraIntrinsics",
    "DenseSample",
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
