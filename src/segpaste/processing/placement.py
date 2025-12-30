"""Object placement utilities for copy-paste operations."""

import random
from dataclasses import dataclass
from typing import Protocol

import torch
from torchvision.ops import box_iou


@dataclass(slots=True, frozen=True)
class PlacementCandidate:
    """Represents a potential placement position for an object."""

    top: int
    left: int
    object_height: int
    object_width: int

    @property
    def bottom(self) -> int:
        return self.top + self.object_height

    @property
    def right(self) -> int:
        return self.left + self.object_width

    def to_box(self) -> torch.Tensor:
        """Convert to bounding box format [left, top, right, bottom]."""
        return torch.tensor([self.left, self.top, self.right, self.bottom])


@dataclass(slots=True, frozen=True)
class PlacementResult:
    """Result of placing an object."""

    image: torch.Tensor
    mask: torch.Tensor
    box: torch.Tensor
    label: torch.Tensor


def get_random_placement(
    target_height: int,
    target_width: int,
    object_height: int,
    object_width: int,
    margin: int,
) -> tuple[int, int]:
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
    new_box: torch.Tensor, existing_boxes: torch.Tensor, iou_threshold: float
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
    return (ious > iou_threshold).any().item()  # type: ignore[no-any-return]


class PlacementGenerator(Protocol):
    """Protocol for generating placement candidates."""

    def generate_candidate(
        self, object_height: int, object_width: int
    ) -> PlacementCandidate | None:
        """Generate a placement candidate."""
        ...


class PlacementValidator(Protocol):
    """Protocol for validating placement candidates."""

    def is_valid(self, candidate: PlacementCandidate) -> bool:
        """Check if a placement candidate is valid."""
        ...


class RandomPlacementGenerator:
    """Generates random placement candidates within image bounds."""

    def __init__(self, image_height: int, image_width: int, margin: int):
        """Initialize RandomPlacementGenerator.

        Args:
            image_height (int): Height of the target image
            image_width (int): Width of the target image
            margin (int): Margin from image edges. The should be at least this far
                from the edge of the image. Must be non-negative.
        """
        self.image_height = image_height
        self.image_width = image_width
        self.margin = margin

    def generate_candidate(
        self, object_height: int, object_width: int
    ) -> PlacementCandidate | None:
        """Generate a random placement candidate."""
        # Check if object can fit at all
        if (
            object_height + 2 * self.margin > self.image_height
            or object_width + 2 * self.margin > self.image_width
        ):
            return None

        # Use existing get_random_placement function
        top, left = get_random_placement(
            self.image_height,
            self.image_width,
            object_height,
            object_width,
            margin=self.margin,
        )

        return PlacementCandidate(top, left, object_height, object_width)


class PaddingAwarePlacementGenerator:
    """Generates placement candidates that respect padding masks."""

    def __init__(
        self,
        image_height: int,
        image_width: int,
        padding_mask: torch.Tensor,
        margin: int,
    ):
        """Initialize PaddingAwarePlacementGenerator.

        Args:
            image_height (int): Height of the target image
            image_width (int): Width of the target image
            padding_mask (torch.Tensor): (1, H, W) Padding mask indicating valid
                placement areas
            margin (int): Must be non-negative. Margin from valid region edges.
                The placement should be at least this far from the edges
                of the valid region.
        """
        self.image_height = image_height
        self.image_width = image_width
        self.padding_mask = padding_mask
        self.margin = margin

        # Find the bounding box of the valid (non-padded) region
        valid_pixels = (self.padding_mask == 0).squeeze().nonzero(as_tuple=False)

        if valid_pixels.numel() > 0:
            self.valid_top = int(valid_pixels[:, 0].min().item())
            self.valid_bottom = int(valid_pixels[:, 0].max().item()) + 1
            self.valid_left = int(valid_pixels[:, 1].min().item())
            self.valid_right = int(valid_pixels[:, 1].max().item()) + 1
        else:
            # No valid region found, set bounds to prevent placement
            self.valid_top = 0
            self.valid_bottom = 0
            self.valid_left = 0
            self.valid_right = 0

    def generate_candidate(
        self, object_height: int, object_width: int
    ) -> PlacementCandidate | None:
        """Generate a placement candidate that respects padding constraints."""
        # Calculate valid placement bounds within the non-padded region
        min_top = self.valid_top + self.margin
        max_top = min(
            self.valid_bottom - object_height - self.margin,
            self.image_height - object_height - self.margin,
        )

        min_left = self.margin + self.valid_left
        max_left = min(
            self.valid_right - object_width - self.margin,
            self.image_width - object_width - self.margin,
        )

        if max_top <= min_top or max_left <= min_left:
            return None

        # Generate random position within valid bounds
        top = random.randint(min_top, max_top)
        left = random.randint(min_left, max_left)

        return PlacementCandidate(top, left, object_height, object_width)


class BoundsValidator:
    """Validates that placement stays within image bounds."""

    def __init__(self, image_height: int, image_width: int):
        self.image_height = image_height
        self.image_width = image_width

    def is_valid(self, candidate: PlacementCandidate) -> bool:
        """Check if candidate fits within image bounds."""
        return (
            candidate.top >= 0
            and candidate.left >= 0
            and candidate.bottom <= self.image_height
            and candidate.right <= self.image_width
        )


class OverlapValidator:
    """Validates that placement doesn't overlap too much with existing objects."""

    def __init__(self, existing_boxes: list[torch.Tensor], iou_threshold: float):
        self.existing_boxes = existing_boxes
        self.iou_threshold = iou_threshold

    def is_valid(self, candidate: PlacementCandidate) -> bool:
        """Check if candidate overlaps with existing objects less than the threshold.

        If any existing box is overlapping with the candidate box
        strictly more than the IoU threshold, the placement is considered invalid.
        """
        if not self.existing_boxes:
            return True

        candidate_box = candidate.to_box()
        existing_boxes_tensor = torch.stack(self.existing_boxes, dim=0)
        return not check_collision(
            candidate_box, existing_boxes_tensor, self.iou_threshold
        )


class ObjectPlacer:
    """Handles finding valid placements for objects using a composable approach."""

    def __init__(
        self, generator: PlacementGenerator, validators: list[PlacementValidator]
    ):
        self.generator = generator
        self.validators = validators

    def find_valid_placement(
        self, object_height: int, object_width: int, max_attempts: int
    ) -> PlacementCandidate | None:
        """Find a valid placement for an object."""
        for _ in range(max_attempts):
            candidate = self.generator.generate_candidate(object_height, object_width)

            if candidate is None:
                continue

            if self._is_valid_candidate(candidate):
                return candidate

        return None

    def _is_valid_candidate(self, candidate: PlacementCandidate) -> bool:
        """Check if candidate passes all validation rules."""
        return all(validator.is_valid(candidate) for validator in self.validators)


def create_object_placer(
    image_height: int,
    image_width: int,
    existing_boxes: list[torch.Tensor],
    padding_mask: torch.Tensor | None,
    margin: int,
    collision_threshold: float,
) -> ObjectPlacer:
    """Factory function to create an ObjectPlacer with appropriate configuration."""

    # Choose generator based on whether we have a padding mask
    generator: PlacementGenerator
    if padding_mask is not None:
        generator = PaddingAwarePlacementGenerator(
            image_height, image_width, padding_mask, margin
        )
    else:
        generator = RandomPlacementGenerator(image_height, image_width, margin)

    # Set up validators
    validators: list[PlacementValidator] = [
        BoundsValidator(image_height, image_width),
        OverlapValidator(existing_boxes, collision_threshold),
    ]

    return ObjectPlacer(generator, validators)
