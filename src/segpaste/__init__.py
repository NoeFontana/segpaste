# Import main functionality here
try:
    import faster_coco_eval as _faster_coco_eval

    _faster_coco_eval.init_as_pycocotools()
except ImportError:
    pass
from segpaste.augmentation import (
    CopyPasteAugmentation,
    CopyPasteCollator,
    CopyPasteTransform,
)
from segpaste.config import CopyPasteConfig
from segpaste.types import DetectionTarget

__all__ = [
    "CopyPasteAugmentation",
    "CopyPasteConfig",
    "DetectionTarget",
    "CopyPasteTransform",
    "CopyPasteCollator",
]
