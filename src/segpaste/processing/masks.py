"""Mask processing utilities for copy-paste operations."""

import torch


def boxes_to_masks(boxes: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Convert bounding boxes to binary masks.

    Args:
        boxes: Tensor of shape [N, 4] in xyxy format
        height: Image height
        width: Image width

    Returns:
        Binary masks of shape [N, height, width]
    """
    device = boxes.device
    dtype = boxes.dtype

    # Create coordinate grids
    y_coords = torch.arange(height, device=device, dtype=dtype).view(height, 1)
    x_coords = torch.arange(width, device=device, dtype=dtype).view(1, width)

    # Expand for batch processing
    y_grid = y_coords.expand(boxes.shape[0], height, width)
    x_grid = x_coords.expand(boxes.shape[0], height, width)

    # Extract box coordinates
    x1 = boxes[:, 0].view(-1, 1, 1)
    y1 = boxes[:, 1].view(-1, 1, 1)
    x2 = boxes[:, 2].view(-1, 1, 1)
    y2 = boxes[:, 3].view(-1, 1, 1)

    # Create masks
    masks = ((x_grid >= x1) & (x_grid <= x2) & (y_grid >= y1) & (y_grid <= y2)).float()

    return masks


def compute_mask_area(masks: torch.Tensor) -> torch.Tensor:
    """Compute area of masks.

    Args:
        masks: Tensor of shape [N, H, W] with binary masks

    Returns:
        Tensor of shape [N] with area for each mask
    """
    return masks.sum(dim=(1, 2))
