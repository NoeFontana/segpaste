# Import main functionality here
import logging

try:
    import faster_coco_eval as _faster_coco_eval

    _faster_coco_eval.init_as_pycocotools()
except ImportError:
    logging.getLogger(__file__).warning("faster_coco_eval not found.")
    pass
from segpaste.augmentation import (
    CopyPasteCollator,
    CopyPasteTransform,
    FixedSizeCrop,
    RandomResize,
    SanitizeBoundingBoxes,
    make_large_scale_jittering,
)
from segpaste.config import CopyPasteConfig
from segpaste.types import DetectionTarget, PaddingMask

__all__ = [
    "PaddingMask",
    "CopyPasteConfig",
    "DetectionTarget",
    "CopyPasteTransform",
    "CopyPasteCollator",
    "FixedSizeCrop",
    "RandomResize",
    "SanitizeBoundingBoxes",
    "make_large_scale_jittering",
]
