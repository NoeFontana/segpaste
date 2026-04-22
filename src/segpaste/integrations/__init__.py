"""Input/Output utilities for datasets and data loading."""

# labels_getter is demoted to the private `_internal` surface per ADR-0001
# Part (i); `as X` keeps it a re-export for internal callers while absent
# from __all__.
from segpaste.integrations.coco import (
    CocoDetectionV2,
    create_coco_dataloader,
)
from segpaste.integrations.coco import labels_getter as labels_getter

__all__ = [
    "CocoDetectionV2",
    "create_coco_dataloader",
]
