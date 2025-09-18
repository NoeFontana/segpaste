# Import main functionality here
try:
    import faster_coco_eval as _faster_coco_eval

    _faster_coco_eval.init_as_pycocotools()
except ImportError:
    pass


from .copy_paste import CopyPasteAugmentation
from .data_types import CopyPasteConfig, DetectionTarget
from .transforms import CopyPasteCollator, CopyPasteTransform

__all__ = [
    "CopyPasteAugmentation",
    "CopyPasteConfig",
    "DetectionTarget",
    "CopyPasteTransform",
    "CopyPasteCollator",
]
