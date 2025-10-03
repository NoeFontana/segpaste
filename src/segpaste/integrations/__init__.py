"""Input/Output utilities for datasets and data loading."""

from segpaste.integrations.coco import (
    CocoDetectionV2,
    create_coco_dataloader,
    labels_getter,
)

__all__ = [
    "CocoDetectionV2",
    "create_coco_dataloader",
    "labels_getter",
]
