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
    BankSource,
    BatchCopyPaste,
    FixedSizeCrop,
    IntraBatchSource,
    RandomResize,
    SanitizeBoundingBoxes,
    SourceStrategy,
    make_large_scale_jittering,
)
from segpaste.integrations import CocoDetectionV2, create_coco_dataloader
from segpaste.integrations.huggingface import (
    from_hf_format,
    make_hf_collate_fn,
    to_hf_batch,
    to_hf_format,
)
from segpaste.integrations.lightning import make_segpaste_datamodule
from segpaste.integrations.torchvision import make_segpaste_collate_fn
from segpaste.presets import PresetConfig, get_preset, list_presets, register_preset
from segpaste.types import (
    BatchAuditPacket,
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
    "BankSource",
    "BatchAuditPacket",
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
    "from_hf_format",
    "get_preset",
    "list_presets",
    "make_hf_collate_fn",
    "make_large_scale_jittering",
    "make_segpaste_collate_fn",
    "make_segpaste_datamodule",
    "register_preset",
    "to_hf_batch",
    "to_hf_format",
]
