"""Type definitions for the segpaste package.

This module contains all data structures and configuration types used
throughout the package.
"""

from segpaste.types.batched_dense_sample import BatchedDenseSample
from segpaste.types.data_structures import PaddingMask
from segpaste.types.dense_sample import (
    CameraIntrinsics,
    DenseSample,
    InstanceMask,
    Modality,
    PanopticMap,
    PanopticSchema,
    PanopticSchemaSpec,
    SemanticMap,
)
from segpaste.types.padded_batched_dense_sample import PaddedBatchedDenseSample
from segpaste.types.type_aliases import (
    BoxesTensor,
    ImageTensor,
    LabelsTensor,
    MasksTensor,
)

__all__ = [
    "BatchedDenseSample",
    "BoxesTensor",
    "CameraIntrinsics",
    "DenseSample",
    "ImageTensor",
    "InstanceMask",
    "LabelsTensor",
    "MasksTensor",
    "Modality",
    "PaddedBatchedDenseSample",
    "PaddingMask",
    "PanopticMap",
    "PanopticSchema",
    "PanopticSchemaSpec",
    "SemanticMap",
]
