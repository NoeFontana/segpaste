"""Object placement utilities for copy-paste operations."""

from typing import Tuple

import torch
from torchvision.ops import box_iou


def get_random_placement(
    target_height: int,
    target_width: int,
    object_height: int,
    object_width: int,
    margin: int = 0,
) -> Tuple[int, int]:
    """Get random placement coordinates for an object.

    Args:
        target_height: Height of target image
        target_width: Width of target image
        object_height: Height of object to place
        object_width: Width of object to place
        margin: Margin from edges

    Returns:
        Top-left coordinates (y, x) for placement
    """
    max_y = max(0, target_height - object_height - margin)
    max_x = max(0, target_width - object_width - margin)

    y = torch.randint(margin, max_y + 1, (1,)).item() if max_y > margin else margin
    x = torch.randint(margin, max_x + 1, (1,)).item() if max_x > margin else margin

    return int(y), int(x)


def check_collision(
    new_box: torch.Tensor, existing_boxes: torch.Tensor, iou_threshold: float = 0.3
) -> bool:
    """Check if a new box collides with existing boxes.

    Args:
        new_box: New bounding box of shape [4] in xyxy format
        existing_boxes: Existing boxes of shape [N, 4] in xyxy format
        iou_threshold: IoU threshold for collision detection

    Returns:
        True if collision detected
    """
    if existing_boxes.numel() == 0:
        return False

    # Compute IoU
    ious = box_iou(new_box.unsqueeze(0), existing_boxes)
    return (ious > iou_threshold).any().item()  # type: ignore
