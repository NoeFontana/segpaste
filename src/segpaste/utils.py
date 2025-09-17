"""Utility functions for copy-paste augmentation."""

from typing import Tuple

import torch
import torch.nn.functional as F


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
    masks = (x_grid >= x1) & (x_grid <= x2) & (y_grid >= y1) & (y_grid <= y2)

    return masks.float()


def masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    """Convert binary masks to bounding boxes.

    Args:
        masks: Binary masks of shape [N, H, W]

    Returns:
        Bounding boxes of shape [N, 4] in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), dtype=torch.float32, device=masks.device)

    n = masks.shape[0]
    boxes = torch.zeros((n, 4), dtype=torch.float32, device=masks.device)

    for i in range(n):
        mask = masks[i]
        if mask.sum() == 0:
            continue

        # Find nonzero coordinates
        nonzero_indices = mask.nonzero()
        y_coords = nonzero_indices[:, 0]
        x_coords = nonzero_indices[:, 1]

        boxes[i, 0] = x_coords.min().float()  # x1
        boxes[i, 1] = y_coords.min().float()  # y1
        boxes[i, 2] = x_coords.max().float()  # x2
        boxes[i, 3] = y_coords.max().float()  # y2

    return boxes


def compute_box_area(boxes: torch.Tensor) -> torch.Tensor:
    """Compute area of bounding boxes.

    Args:
        boxes: Tensor of shape [N, 4] in xyxy format

    Returns:
        Areas of shape [N]
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def compute_mask_area(masks: torch.Tensor) -> torch.Tensor:
    """Compute area of binary masks.

    Args:
        masks: Binary masks of shape [N, H, W]

    Returns:
        Areas of shape [N]
    """
    return masks.sum(dim=(1, 2))


def random_flip_horizontal(
    image: torch.Tensor,
    boxes: torch.Tensor,
    masks: torch.Tensor | None = None,
    probability: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Randomly flip image, boxes and masks horizontally.

    Args:
        image: Image tensor of shape [C, H, W]
        boxes: Bounding boxes of shape [N, 4] in xyxy format
        masks: Optional masks of shape [N, H, W]
        probability: Probability of flipping

    Returns:
        Flipped image, boxes, and masks
    """
    if torch.rand(1).item() < probability:
        # Flip image
        image = torch.flip(image, dims=[2])

        # Flip boxes
        img_width = image.shape[2]
        boxes_flipped = boxes.clone()
        boxes_flipped[:, 0] = img_width - boxes[:, 2]  # x1 = W - x2
        boxes_flipped[:, 2] = img_width - boxes[:, 0]  # x2 = W - x1

        # Flip masks
        if masks is not None:
            masks = torch.flip(masks, dims=[2])

        return image, boxes_flipped, masks

    return image, boxes, masks


def resize_image_and_targets(
    image: torch.Tensor,
    boxes: torch.Tensor,
    masks: torch.Tensor | None,
    target_size: Tuple[int, int],
    keep_aspect_ratio: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, Tuple[float, float]]:
    """Resize image and scale boxes/masks accordingly.

    Args:
        image: Image tensor of shape [C, H, W]
        boxes: Bounding boxes of shape [N, 4] in xyxy format
        masks: Optional masks of shape [N, H, W]
        target_size: Target (height, width)
        keep_aspect_ratio: Whether to maintain aspect ratio

    Returns:
        Resized image, scaled boxes, scaled masks, and scale factors
    """
    original_h, original_w = image.shape[1], image.shape[2]
    target_h, target_w = target_size

    if keep_aspect_ratio:
        scale = min(target_h / original_h, target_w / original_w)
        new_h = int(original_h * scale)
        new_w = int(original_w * scale)
    else:
        scale_h = target_h / original_h
        scale_w = target_w / original_w
        new_h, new_w = target_h, target_w
        scale = (scale_h, scale_w)

    # Resize image
    resized_image = F.interpolate(
        image.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False
    ).squeeze(0)

    # Scale boxes
    if keep_aspect_ratio:
        scale_factors = (scale, scale)
        scaled_boxes = boxes * scale
    else:
        scale_factors = (scale_h, scale_w)
        scaled_boxes = boxes.clone()
        scaled_boxes[:, [0, 2]] *= scale_w  # x coordinates
        scaled_boxes[:, [1, 3]] *= scale_h  # y coordinates

    # Scale masks
    scaled_masks = None
    if masks is not None:
        scaled_masks = F.interpolate(
            masks.unsqueeze(1).float(),
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        scaled_masks = (scaled_masks > 0.5).float()

    return resized_image, scaled_boxes, scaled_masks, scale_factors


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

    return y, x


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
    ious = compute_iou(new_box.unsqueeze(0), existing_boxes)
    return (ious > iou_threshold).any().item()


def compute_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute IoU between two sets of boxes.

    Args:
        boxes1: First set of boxes [N, 4] in xyxy format
        boxes2: Second set of boxes [M, 4] in xyxy format

    Returns:
        IoU matrix of shape [N, M]
    """
    # Compute intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    intersection = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    # Compute union
    area1 = compute_box_area(boxes1)  # [N]
    area2 = compute_box_area(boxes2)  # [M]
    union = area1[:, None] + area2 - intersection

    # Avoid division by zero
    iou = intersection / (union + 1e-8)
    return iou
