"""Input/Output utilities for datasets and data loading."""

from segpaste.integrations.coco import (
    FilteredCocoDetection,
    create_coco_dataset,
    labels_getter,
)

__all__ = [
    "FilteredCocoDetection",
    "create_coco_dataset",
    "labels_getter",
]
