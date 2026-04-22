"""Copy-paste augmentation implementation for PyTorch."""

import random

import torch
from torchvision import tv_tensors
from torchvision.ops import masks_to_boxes

from segpaste.config import CopyPasteConfig
from segpaste.processing import (
    compute_mask_area,
)
from segpaste.processing.placement import PlacementResult, create_object_placer
from segpaste.types import DenseSample, InstanceMask


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
        target_data: DenseSample,
        source_objects: list[DenseSample],
    ) -> DenseSample:
        """Apply copy-paste augmentation to target image."""
        if target_data.instance_masks is None or target_data.instance_ids is None:
            raise ValueError("INSTANCE modality required on target_data")

        if not self._should_apply_augmentation(source_objects):
            return target_data

        selected_objects = self._select_objects_to_paste(source_objects)
        if not selected_objects:
            return target_data

        return self._apply_copy_paste(target_data, selected_objects)

    def _should_apply_augmentation(self, source_objects: list[DenseSample]) -> bool:
        """Check if augmentation should be applied."""
        return (
            len(source_objects) > 0 and random.random() <= self.config.paste_probability
        )

    def _select_objects_to_paste(
        self, source_objects: list[DenseSample]
    ) -> list[DenseSample]:
        """Select random objects to paste."""
        if not source_objects:
            return []

        num_available = len(source_objects)
        max_paste = min(self.config.max_paste_objects, num_available)
        min_paste = min(self.config.min_paste_objects, max_paste)

        num_to_paste = random.randint(min_paste, max_paste)
        return random.sample(source_objects, num_to_paste)

    def _apply_copy_paste(
        self, target_data: DenseSample, paste_objects: list[DenseSample]
    ) -> DenseSample:
        """Apply copy-paste augmentation to target image."""
        # Unwrap the tv_tensor image into a plain Tensor so in-place arithmetic
        # does not trip tv_tensor dispatch during blending; we re-wrap at the
        # final DenseSample construction in ``_update_annotations``.
        image = target_data.image.clone().as_subclass(torch.Tensor)
        padding_mask = (
            target_data.padding_mask.as_subclass(torch.Tensor)
            if target_data.padding_mask is not None
            else None
        )

        pasted_results = self._paste_all_objects(image, padding_mask, paste_objects)

        if not pasted_results:
            return target_data

        return self._update_annotations(target_data, pasted_results)

    def _paste_all_objects(
        self,
        image: torch.Tensor,
        padding_mask: torch.Tensor | None,
        paste_objects: list[DenseSample],
    ) -> list[PlacementResult]:
        """Paste all objects onto the image and return successful placements."""
        pasted_results: list[PlacementResult] = []
        pasted_boxes: list[torch.Tensor] = []

        target_size = (image.shape[1], image.shape[2])  # (H, W)
        if padding_mask is not None and padding_mask.shape[1:] != image.shape[1:]:
            raise ValueError("Padding mask shape mismatch")

        for obj in paste_objects:
            if not self._is_valid_object(obj):
                continue
            # Narrow instance_masks locally (_is_valid_object has already verified).
            obj_masks = obj.instance_masks
            if obj_masks is None:
                continue

            for i in range(obj_masks.shape[0]):
                placement = self._try_place_single_object(
                    image, obj, i, target_size, pasted_boxes, padding_mask
                )

                if placement:
                    image = placement.image
                    pasted_boxes.append(placement.box.squeeze(0))
                    pasted_results.append(placement)

        return pasted_results

    def _is_valid_object(self, obj: DenseSample) -> bool:
        """Check if object is valid for pasting."""
        if obj.instance_masks is None or obj.boxes.shape[0] == 0:
            return False

        box_h, box_w = (
            obj.boxes[:, 3] - obj.boxes[:, 1],
            obj.boxes[:, 2] - obj.boxes[:, 0],
        )

        # Object must not be too thin
        min_edge = min(box_h.min().item(), box_w.min().item())
        if min_edge < self.config.min_object_edge:
            return False
        # Object pixel count must be sufficient
        return (
            not (obj.instance_masks.sum(dim=(1, 2)) < self.config.min_object_area)
            .any()
            .item()
        )

    def _try_place_single_object(
        self,
        image: torch.Tensor,
        obj: DenseSample,
        obj_idx: int,
        target_size: tuple[int, int],
        existing_boxes: list[torch.Tensor],
        padding_mask: torch.Tensor | None,
    ) -> PlacementResult | None:
        """Try to place a single object on the image."""
        if obj.instance_masks is None:
            return None
        # Extract object components (keep batch dimension for consistency)
        obj_mask = obj.instance_masks[obj_idx : obj_idx + 1]
        obj_box = obj.boxes[obj_idx : obj_idx + 1]
        obj_label = obj.labels[obj_idx : obj_idx + 1]

        # Crop source image and mask to bounding box region
        x1, y1, x2, y2 = obj_box[0].int()

        # Ensure coordinates are within image boundaries
        img_h, img_w = obj.image.shape[1], obj.image.shape[2]
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(x1 + 1, min(x2, img_w))
        y2 = max(y1 + 1, min(y2, img_h))

        # Drop the tv_tensor wrapper for arithmetic on the crop.
        source_image = obj.image.as_subclass(torch.Tensor)[:, y1:y2, x1:x2]
        cropped_mask = obj_mask[:, y1:y2, x1:x2]

        # Try placement
        result = self._place_object(
            image,
            source_image,
            cropped_mask,
            obj_box,
            obj_label,
            target_size,
            existing_boxes,
            max_attempts=10,
            padding_mask=padding_mask,
        )

        return result

    def _update_annotations(
        self, target_data: DenseSample, pasted_results: list[PlacementResult]
    ) -> DenseSample:
        """Update target annotations with pasted objects."""
        if target_data.instance_masks is None or target_data.instance_ids is None:
            raise ValueError("_update_annotations requires INSTANCE modality")

        # Extract pasted data
        final_image = pasted_results[-1].image  # Use final image state
        pasted_masks = torch.stack(
            [r.mask.squeeze(0).to(torch.bool) for r in pasted_results]
        )
        pasted_boxes = torch.stack([r.box.squeeze(0) for r in pasted_results])
        pasted_labels = torch.stack([r.label.squeeze(0) for r in pasted_results])

        # Update existing objects for occlusion
        original_masks = target_data.instance_masks.as_subclass(torch.Tensor).to(
            torch.bool
        )
        updated_masks, updated_boxes, updated_labels, valid_indices = (
            self._update_and_filter_occluded_objects(
                original_masks,
                target_data.boxes,
                target_data.labels,
                pasted_masks,
            )
        )

        # Survivor instance_ids + fresh ids for each newly pasted object.
        survivor_ids = target_data.instance_ids[valid_indices]
        max_prev = (
            int(target_data.instance_ids.max().item())
            if target_data.instance_ids.numel() > 0
            else -1
        )
        k = pasted_masks.shape[0]
        new_ids = torch.arange(
            max_prev + 1,
            max_prev + 1 + k,
            dtype=torch.int32,
            device=survivor_ids.device,
        )

        # Combine all annotations
        all_masks = torch.cat([updated_masks, pasted_masks], dim=0)
        all_boxes = torch.cat([updated_boxes, pasted_boxes], dim=0)
        all_labels = torch.cat([updated_labels, pasted_labels], dim=0)
        all_ids = torch.cat([survivor_ids, new_ids], dim=0)

        h, w = final_image.shape[-2:]
        return DenseSample(
            image=tv_tensors.Image(final_image),
            boxes=tv_tensors.BoundingBoxes(  # pyright: ignore[reportCallIssue]
                all_boxes,
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=(h, w),
            ),
            labels=all_labels,
            instance_ids=all_ids,
            instance_masks=InstanceMask(all_masks),
            padding_mask=target_data.padding_mask,
        )

    def _place_object(
        self,
        target_image: torch.Tensor,
        source_image: torch.Tensor,
        source_mask: torch.Tensor,
        source_box: torch.Tensor,
        source_label: torch.Tensor,
        target_size: tuple[int, int],
        existing_boxes: list[torch.Tensor],
        max_attempts: int,
        padding_mask: torch.Tensor | None,
    ) -> PlacementResult | None:
        """Place an object on the target image."""
        obj_h = int(source_box[0, 3] - source_box[0, 1])
        obj_w = int(source_box[0, 2] - source_box[0, 0])

        # Try to find valid placement position
        placement_pos = self._find_valid_placement(
            target_size,
            (obj_h, obj_w),
            existing_boxes,
            padding_mask,
            max_attempts,
        )
        if placement_pos is None:
            return None

        top, left = placement_pos

        # Create placed objects
        placed_image = self._blend_object_on_target(
            target_image, source_image, source_mask, top, left
        )
        placed_mask = self._create_placed_mask(source_mask, target_size, top, left)
        placed_box = self._create_placed_box(source_box, left, top, obj_w, obj_h)

        return PlacementResult(placed_image, placed_mask, placed_box, source_label)

    def _find_valid_placement(
        self,
        target_size: tuple[int, int],
        object_size: tuple[int, int],
        existing_boxes: list[torch.Tensor],
        padding_mask: torch.Tensor | None,
        max_attempts: int,
    ) -> tuple[int, int] | None:
        """Find a valid placement position for the object."""
        target_h, target_w = target_size
        obj_h, obj_w = object_size

        placer = create_object_placer(
            image_height=target_h,
            image_width=target_w,
            existing_boxes=existing_boxes,
            padding_mask=padding_mask,
            margin=0,
            collision_threshold=0.9,
        )

        candidate = placer.find_valid_placement(obj_h, obj_w, max_attempts)

        if candidate is None:
            return None

        return candidate.top, candidate.left

    def _create_placed_mask(
        self,
        source_mask: torch.Tensor,
        target_size: tuple[int, int],
        top: int,
        left: int,
    ) -> torch.Tensor:
        """Create mask for placed object in target coordinates."""
        target_h, target_w = target_size
        source_h, source_w = source_mask.shape[1], source_mask.shape[2]

        placed_mask = torch.zeros(
            1, target_h, target_w, dtype=source_mask.dtype, device=source_mask.device
        )

        # Ensure we don't go beyond target boundaries
        actual_h = min(source_h, target_h - top)
        actual_w = min(source_w, target_w - left)

        if actual_h > 0 and actual_w > 0:
            placed_mask[0, top : top + actual_h, left : left + actual_w] = source_mask[
                0, :actual_h, :actual_w
            ]

        return placed_mask

    def _create_placed_box(
        self, source_box: torch.Tensor, left: int, top: int, width: int, height: int
    ) -> torch.Tensor:
        """Create bounding box for placed object."""
        return torch.tensor(
            [[left, top, left + width, top + height]],
            dtype=source_box.dtype,
            device=source_box.device,
        )

    def _blend_object_on_target(
        self,
        target_image: torch.Tensor,
        source_image: torch.Tensor,
        source_mask: torch.Tensor,
        top: int,
        left: int,
    ) -> torch.Tensor:
        """Blend source object onto target image using simple alpha blending."""
        result_image = target_image.clone()
        source_h, source_w = source_image.shape[1], source_image.shape[2]

        # Extract target region - make sure it doesn't go beyond image bounds
        target_h, target_w = target_image.shape[1], target_image.shape[2]

        # Clamp the region to fit within target image
        actual_h = min(source_h, target_h - top)
        actual_w = min(source_w, target_w - left)

        if actual_h <= 0 or actual_w <= 0:
            return result_image  # Nothing to blend

        target_region = target_image[:, top : top + actual_h, left : left + actual_w]

        # Crop source image and mask to match the actual region size
        cropped_source = source_image[:, :actual_h, :actual_w]
        cropped_mask = source_mask[:, :actual_h, :actual_w].to(target_image.dtype)

        # Get the mask and ensure it matches the source image dimensions
        mask = cropped_mask[0]  # Remove the first dimension

        # Ensure mask has proper dimensions for broadcasting
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)  # Add channel dimension

        # Simple alpha blending: target * (1 - mask) + source * mask
        alpha = 1.0
        blended_region = target_region * (1.0 - alpha * mask) + cropped_source * (
            alpha * mask
        )

        result_image[:, top : top + actual_h, left : left + actual_w] = blended_region

        return result_image

    def _update_and_filter_occluded_objects(
        self,
        original_masks: torch.Tensor,
        original_boxes: torch.Tensor,
        original_labels: torch.Tensor,
        pasted_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update masks for occlusion and filter heavily occluded objects.

        Returns ``(filtered_masks, filtered_boxes, filtered_labels, valid_indices)``
        — ``valid_indices`` is the ``int64`` index tensor into the original N used
        by the caller to slice co-indexed fields like ``instance_ids``.
        """
        if original_masks.numel() == 0:
            empty_indices = torch.empty(
                (0,), dtype=torch.int64, device=original_masks.device
            )
            return original_masks, original_boxes, original_labels, empty_indices

        # Remove occluded parts from original masks (bool arithmetic).
        occlusion_mask = pasted_masks.any(dim=0)
        updated_masks = original_masks & ~occlusion_mask

        # Filter objects based on remaining area
        valid_indices = self._find_valid_objects(original_masks, updated_masks)

        if valid_indices.numel() == 0:
            spatial_shape = (original_masks.shape[1], original_masks.shape[2])
            empty_masks, empty_boxes, empty_labels = self._create_empty_annotations(
                spatial_shape,
                original_masks.device,
                original_masks.dtype,
                original_boxes.dtype,
                original_labels.dtype,
            )
            empty_indices = torch.empty(
                (0,), dtype=torch.int64, device=original_masks.device
            )
            return empty_masks, empty_boxes, empty_labels, empty_indices

        # Keep only valid objects and recompute boxes
        filtered_masks = updated_masks[valid_indices]
        filtered_labels = original_labels[valid_indices]
        filtered_boxes = masks_to_boxes(filtered_masks)

        return filtered_masks, filtered_boxes, filtered_labels, valid_indices

    def _find_valid_objects(
        self, original_masks: torch.Tensor, updated_masks: torch.Tensor
    ) -> torch.Tensor:
        """Find objects that are not too heavily occluded."""
        original_areas = compute_mask_area(original_masks)
        updated_areas = compute_mask_area(updated_masks)

        # Objects must have remaining area
        has_area = updated_areas > 0

        # Objects must not be too occluded
        occlusion_ratios = 1.0 - (updated_areas / (original_areas + 1e-8))
        not_too_occluded = occlusion_ratios <= self.config.occluded_area_threshold

        return torch.where(has_area & not_too_occluded)[0]

    def _create_empty_annotations(
        self,
        spatial_shape: tuple[int, int],
        device: torch.device,
        mask_dtype: torch.dtype,
        box_dtype: torch.dtype,
        label_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create empty annotations when no objects remain."""
        img_height, img_width = spatial_shape
        empty_masks = torch.empty(
            (0, img_height, img_width), dtype=mask_dtype, device=device
        )
        empty_boxes = torch.empty((0, 4), dtype=box_dtype, device=device)
        empty_labels = torch.empty((0,), dtype=label_dtype, device=device)
        return empty_masks, empty_boxes, empty_labels
