"""Copy-paste augmentation implementation for PyTorch."""

import random
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torchvision.ops import masks_to_boxes

from segpaste.blending import blend_with_mode, create_smooth_mask_border
from segpaste.data_types import CopyPasteConfig, DetectionTarget
from segpaste.utils import (
    check_collision,
    compute_mask_area,
    get_random_placement,
)


class CopyPasteAugmentation:
    """Copy-paste augmentation for instance segmentation and object detection.

    This implementation follows the approach described in:
    "Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation"
    https://arxiv.org/abs/2012.07177
    """

    def __init__(self, config: CopyPasteConfig):
        """Initialize copy-paste augmentation.

        Args:
            config: Configuration for copy-paste augmentation
        """
        self.config = config

    def transform(
        self,
        target_data: DetectionTarget,
        source_objects: List[DetectionTarget],
    ) -> DetectionTarget:
        """Apply copy-paste augmentation to target image.

        Args:
            target_data: Target image and annotations
            source_objects: List of source objects to potentially paste

        Returns:
            Augmented target data
        """
        # Check if we should apply copy-paste
        if not source_objects or target_data.masks is None:
            return target_data

        if random.random() > self.config.paste_probability:
            return target_data

        selected_objects = self._select_objects_to_paste(source_objects)
        if not selected_objects:
            return target_data

        return self._apply_copy_paste(target_data, selected_objects)

    def _select_objects_to_paste(
        self, source_objects: List[DetectionTarget]
    ) -> List[DetectionTarget]:
        """Select objects to paste from source objects.

        Args:
            source_objects: List of available source objects

        Returns:
            List of selected objects to paste
        """
        if not source_objects:
            return []

        max_objects = min(self.config.max_paste_objects, len(source_objects))
        min_objects = min(self.config.min_paste_objects, max_objects)
        num_to_paste = random.randint(min_objects, max_objects)

        selected = random.sample(source_objects, num_to_paste)

        return selected

    def _apply_copy_paste(
        self, target_data: DetectionTarget, paste_objects: List[DetectionTarget]
    ) -> DetectionTarget:
        """Apply copy-paste augmentation to target image.

        Args:
            target_data: Target image and annotations
            paste_objects: Objects to paste

        Returns:
            Augmented target data
        """
        # Work with copies
        image = target_data.image.clone()
        boxes = target_data.boxes.clone()
        labels = target_data.labels.clone()
        masks = target_data.masks.clone() if target_data.masks is not None else None

        if masks is None:
            return target_data

        _, img_height, img_width = image.shape

        # Keep track of pasted boxes for collision detection
        pasted_boxes: list[torch.Tensor] = []
        pasted_masks_list: list[torch.Tensor] = []
        pasted_labels_list: list[torch.Tensor] = []

        for obj in paste_objects:
            if obj.masks is None or obj.masks.numel() == 0:
                continue

            # Process each object in the source (could be multiple objects)
            for obj_idx in range(obj.masks.shape[0]):
                obj_mask = obj.masks[obj_idx : obj_idx + 1]  # Keep batch dim
                obj_box = obj.boxes[obj_idx : obj_idx + 1]
                obj_label = obj.labels[obj_idx : obj_idx + 1]
                obj_img = obj.image

                # Try to place the object
                placed_data = self._place_object(
                    image,
                    obj_img,
                    obj_mask,
                    obj_box,
                    obj_label,
                    (img_height, img_width),
                    pasted_boxes,
                )

                if placed_data is not None:
                    placed_img, placed_mask, placed_box, placed_label = placed_data

                    # Update target image
                    image = placed_img

                    # Track pasted objects
                    pasted_boxes.append(placed_box.squeeze(0))
                    pasted_masks_list.append(placed_mask.squeeze(0))
                    pasted_labels_list.append(placed_label.squeeze(0))

        # Update annotations
        if pasted_masks_list:
            # Add pasted objects to annotations
            pasted_masks = torch.stack(pasted_masks_list, dim=0)
            pasted_boxes_tensor = torch.stack(pasted_boxes, dim=0)
            pasted_labels = torch.stack(pasted_labels_list, dim=0)

            # Update existing objects for occlusion and filter out heavily occluded ones
            masks, boxes, labels = self._update_and_filter_occluded_objects(
                masks, boxes, labels, pasted_masks
            )

            # Concatenate with pasted objects
            all_boxes = torch.cat([boxes, pasted_boxes_tensor], dim=0)
            all_labels = torch.cat([labels, pasted_labels], dim=0)
            all_masks = torch.cat([masks, pasted_masks], dim=0)
        else:
            all_boxes = boxes
            all_labels = labels
            all_masks = masks

        return DetectionTarget(
            image=image, boxes=all_boxes, labels=all_labels, masks=all_masks
        )

    def _place_object(
        self,
        target_image: torch.Tensor,
        source_image: torch.Tensor,
        source_mask: torch.Tensor,
        source_box: torch.Tensor,
        source_label: torch.Tensor,
        target_size: Tuple[int, int],
        existing_boxes: List[torch.Tensor],
        max_attempts: int = 10,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Place an object on the target image.

        Args:
            target_image: Target image [C, H, W]
            source_image: Source image [C, H, W]
            source_mask: Source mask [1, H, W]
            source_box: Source bounding box [1, 4]
            source_label: Source label [1]
            target_size: Target image size (H, W)
            existing_boxes: List of already placed boxes
            max_attempts: Maximum placement attempts

        Returns:
            Tuple of (updated_image, placed_mask, placed_box, placed_label) or None
        """
        target_h, target_w = target_size
        source_h, source_w = source_image.shape[1], source_image.shape[2]

        # Apply random scaling
        scale_min, scale_max = self.config.scale_range
        scale_factor = torch.rand(1).item() * (scale_max - scale_min) + scale_min

        if scale_factor != 1.0:
            new_h = int(source_h * scale_factor)
            new_w = int(source_w * scale_factor)

            if new_h <= 0 or new_w <= 0 or new_h > target_h or new_w > target_w:
                return None

            # Resize source image and mask
            source_image = F.interpolate(
                source_image.unsqueeze(0),
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            source_mask = F.interpolate(
                source_mask.unsqueeze(0),
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            source_mask = (source_mask > 0.5).float()

            source_h, source_w = new_h, new_w

        # Apply random horizontal flip
        if self.config.enable_flip and torch.rand(1).item() < 0.5:
            source_image = torch.flip(source_image, dims=[2])
            source_mask = torch.flip(source_mask, dims=[2])

        # Try to find valid placement
        for _ in range(max_attempts):
            # Get random placement
            top, left = get_random_placement(
                target_h, target_w, source_h, source_w, margin=10
            )

            # Check bounds
            if top + source_h > target_h or left + source_w > target_w:
                continue

            # Create new box for placed object
            new_box = torch.tensor(
                [[left, top, left + source_w, top + source_h]],
                dtype=source_box.dtype,
                device=source_box.device,
            )

            # Check collision with existing boxes
            if existing_boxes:
                existing_boxes_tensor = torch.stack(existing_boxes, dim=0)
                if check_collision(
                    new_box[0], existing_boxes_tensor, iou_threshold=0.1
                ):
                    continue

            # Place the object
            placed_image = self._blend_object_on_target(
                target_image, source_image, source_mask, top, left
            )

            # Create mask for placed object in target coordinates
            placed_mask = torch.zeros(
                1,
                target_h,
                target_w,
                dtype=source_mask.dtype,
                device=source_mask.device,
            )
            placed_mask[0, top : top + source_h, left : left + source_w] = source_mask[
                0
            ]

            return placed_image, placed_mask, new_box, source_label

        return None

    def _blend_object_on_target(
        self,
        target_image: torch.Tensor,
        source_image: torch.Tensor,
        source_mask: torch.Tensor,
        top: int,
        left: int,
    ) -> torch.Tensor:
        """Blend source object onto target image.

        Args:
            target_image: Target image [C, H, W]
            source_image: Source image/patch [C, H_src, W_src]
            source_mask: Source mask [1, H_src, W_src]
            top: Top coordinate for placement
            left: Left coordinate for placement

        Returns:
            Blended target image
        """
        result_image = target_image.clone()
        source_h, source_w = source_image.shape[1], source_image.shape[2]

        # Extract target region
        target_region = target_image[:, top : top + source_h, left : left + source_w]

        # Create smooth mask for better blending
        smooth_mask = create_smooth_mask_border(
            source_mask[0], border_width=3, sigma=1.0
        )

        # Blend source onto target region
        blended_region = blend_with_mode(
            source_image,
            target_region,
            smooth_mask,
            mode=self.config.blend_mode,
            alpha=0.9,
            sigma=1.5,
        )

        # Update result image
        result_image[:, top : top + source_h, left : left + source_w] = blended_region

        return result_image

    def _update_and_filter_occluded_objects(
        self,
        original_masks: torch.Tensor,
        original_boxes: torch.Tensor,
        original_labels: torch.Tensor,
        pasted_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update masks based on occlusion and filter heavily occluded objects.

        Args:
            original_masks: Original object masks [N, H, W]
            original_boxes: Original bounding boxes [N, 4]
            original_labels: Original labels [N]
            pasted_masks: Pasted object masks [M, H, W]

        Returns:
            Tuple of (filtered_masks, filtered_boxes, filtered_labels)
        """
        if original_masks.numel() == 0:
            return original_masks, original_boxes, original_labels

        # Combine all pasted masks to create occlusion mask
        occlusion_mask = pasted_masks.sum(dim=0) > 0  # [H, W]

        # Update original masks by removing occluded parts
        updated_masks = original_masks.clone()
        for i in range(len(original_masks)):
            # Remove occluded pixels
            updated_masks[i] = updated_masks[i] * (~occlusion_mask).float()

        # Compute areas for filtering
        original_areas = compute_mask_area(original_masks)
        updated_areas = compute_mask_area(updated_masks)

        # Find valid masks (not completely occluded and below occlusion threshold)
        valid_mask = updated_areas > 0  # Must have some area left

        # Also check occlusion ratio for objects that still have area
        occlusion_ratios = 1.0 - (updated_areas / (original_areas + 1e-8))
        occlusion_valid = occlusion_ratios <= self.config.occluded_area_threshold

        # Combine both conditions
        valid_indices = torch.where(valid_mask & occlusion_valid)[0]

        if valid_indices.numel() == 0:
            # No valid objects remain
            img_height, img_width = original_masks.shape[1], original_masks.shape[2]
            empty_masks = torch.empty(
                (0, img_height, img_width),
                dtype=original_masks.dtype,
                device=original_masks.device,
            )
            empty_boxes = torch.empty(
                (0, 4), dtype=original_boxes.dtype, device=original_boxes.device
            )
            empty_labels = torch.empty(
                (0,), dtype=original_labels.dtype, device=original_labels.device
            )
            return empty_masks, empty_boxes, empty_labels

        # Filter to keep only valid objects
        filtered_masks = updated_masks[valid_indices]
        filtered_labels = original_labels[valid_indices]

        # Recompute bounding boxes from updated masks (only for valid objects)
        filtered_boxes = masks_to_boxes(filtered_masks)

        return filtered_masks, filtered_boxes, filtered_labels
