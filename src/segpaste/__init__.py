# Import main functionality here
import logging

try:
    import faster_coco_eval as _faster_coco_eval

    _faster_coco_eval.init_as_pycocotools()
except ImportError:
    logging.getLogger(__file__).warning("faster_coco_eval not found.")
    pass
from segpaste.augmentation import (
    CopyPasteAugmentation,
    CopyPasteCollator,
    CopyPasteTransform,
    FixedSizeCrop,
    RandomResize,
    SanitizeBoundingBoxes,
    make_large_scale_jittering,
)
from segpaste.config import CopyPasteConfig
from segpaste.integrations import CocoDetectionV2
from segpaste.types import (
    CameraIntrinsics,
    DenseSample,
    DetectionTarget,
    InstanceMask,
    Modality,
    PaddingMask,
    PanopticMap,
    PanopticSchema,
    SemanticMap,
)

__all__ = [
    "CameraIntrinsics",
    "CocoDetectionV2",
    "CopyPasteAugmentation",
    "CopyPasteCollator",
    "CopyPasteConfig",
    "CopyPasteTransform",
    "DenseSample",
    "DetectionTarget",
    "FixedSizeCrop",
    "InstanceMask",
    "Modality",
    "PaddingMask",
    "PanopticMap",
    "PanopticSchema",
    "RandomResize",
    "SanitizeBoundingBoxes",
    "SemanticMap",
    "make_large_scale_jittering",
]
