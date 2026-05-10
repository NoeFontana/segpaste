# Public top-level surface. Pinned by `tests/test_public_surface.py`; adding
# or removing a name here without amending ADR-0001 / ADR-0003 fails CI.
import logging
from importlib.metadata import version as _pkg_version

try:
    import faster_coco_eval as _faster_coco_eval

    _faster_coco_eval.init_as_pycocotools()
except ImportError:
    logging.getLogger(__file__).warning("faster_coco_eval not found.")

from segpaste._internal.bank import InstanceBank
from segpaste.augmentation import (
    BatchCopyPaste,
    FixedSizeCrop,
    IntraBatchSource,
    RandomResize,
    SanitizeBoundingBoxes,
    SourceStrategy,
    make_large_scale_jittering,
)
from segpaste.integrations import CocoDetectionV2, create_coco_dataloader
from segpaste.presets import PresetConfig, get_preset, list_presets, register_preset
from segpaste.types import (
    BatchedDenseSample,
    CameraIntrinsics,
    DenseSample,
    InstanceMask,
    Modality,
    PaddedBatchedDenseSample,
    PaddingMask,
    PanopticMap,
    PanopticSchema,
    SemanticMap,
)

__version__: str = _pkg_version("segpaste")

__all__ = [
    "BatchCopyPaste",
    "BatchedDenseSample",
    "CameraIntrinsics",
    "CocoDetectionV2",
    "DenseSample",
    "FixedSizeCrop",
    "InstanceBank",
    "InstanceMask",
    "IntraBatchSource",
    "Modality",
    "PaddedBatchedDenseSample",
    "PaddingMask",
    "PanopticMap",
    "PanopticSchema",
    "PresetConfig",
    "RandomResize",
    "SanitizeBoundingBoxes",
    "SemanticMap",
    "SourceStrategy",
    "__version__",
    "create_coco_dataloader",
    "get_preset",
    "list_presets",
    "make_large_scale_jittering",
    "register_preset",
]
