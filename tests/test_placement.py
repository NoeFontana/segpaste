import random

import pytest
import torch

from segpaste.processing.placement import (
    OverlapValidator,
    PaddingAwarePlacementGenerator,
    PlacementCandidate,
)


class TestOverlapValidator:
    """Test cases for OverlapValidator."""

    def test_no_collision_with_empty_boxes(self) -> None:
        """Test placement with no existing boxes."""
        validator = OverlapValidator(existing_boxes=[], iou_threshold=0.5)

        candidate = PlacementCandidate(
            top=10, left=10, object_height=20, object_width=20
        )
        assert validator.is_valid(candidate)

    def test_collision_detection_overlap(self) -> None:
        """Test collision detection with overlapping boxes."""
        existing_boxes = [torch.tensor([10, 10, 30, 30])]  # Box at (10,10) to (30,30)
        validator = OverlapValidator(existing_boxes, iou_threshold=0.1)

        # Overlapping candidate should be invalid
        overlapping_candidate = PlacementCandidate(
            top=20, left=20, object_height=20, object_width=20
        )
        assert not validator.is_valid(overlapping_candidate)

        # Non-overlapping candidate should be valid
        separate_candidate = PlacementCandidate(
            top=50, left=50, object_height=20, object_width=20
        )
        assert validator.is_valid(separate_candidate)

    def test_collision_threshold_sensitivity(self) -> None:
        """Test collision detection with different IoU thresholds."""
        existing_boxes = [torch.tensor([10, 10, 30, 30])]

        # Small overlap candidate
        candidate = PlacementCandidate(
            top=25, left=25, object_height=10, object_width=10
        )

        # Strict threshold should reject small overlaps
        strict_validator = OverlapValidator(existing_boxes, iou_threshold=0.01)
        assert not strict_validator.is_valid(candidate)

        # Lenient threshold might allow small overlaps
        lenient_validator = OverlapValidator(existing_boxes, iou_threshold=0.9)
        assert lenient_validator.is_valid(candidate)

    @pytest.mark.parametrize(
        "iou_threshold,expected_valid",
        [
            (0.1, False),  # IoU ~0.14 > 0.1, invalid
            (0.15, True),  # IoU ~0.14 < 0.15, valid
            (0.5, True),  # IoU ~0.14 < 0.5, valid
        ],
    )
    def test_iou_thresholds(self, iou_threshold: float, expected_valid: bool) -> None:
        """Test OverlapValidator with different IoU thresholds."""
        # Existing box: [10, 10, 30, 30] (20x20)
        # Candidate box: [20, 20, 40, 40] (20x20)
        # Overlap area: [20, 20, 30, 30] (10x10 = 100)
        # Union area: 400 + 400 - 100 = 700
        # IoU = 100/700 ≈ 0.143
        existing_boxes = [torch.tensor([10, 10, 30, 30])]
        validator = OverlapValidator(existing_boxes, iou_threshold)

        candidate = PlacementCandidate(
            top=20, left=20, object_height=20, object_width=20
        )
        assert validator.is_valid(candidate) == expected_valid


class TestPaddingAwarePlacementGenerator:
    """Test cases for PaddingAwarePlacementGenerator."""

    @pytest.mark.parametrize("margin", [0, 5])
    def test_placement_respects_padding_bounds(self, margin: int) -> None:
        """Test that generated candidates respect padding mask bounds."""
        # Create padding mask with valid region in center
        padding_mask = torch.ones(100, 100, dtype=torch.uint8)  # All padded

        min_non_padded_left, min_non_padded_top = 20, 25
        max_non_padded = 80
        padding_mask[
            min_non_padded_top:max_non_padded, min_non_padded_left:max_non_padded
        ] = 0  # Center 60x60 is valid

        generator = PaddingAwarePlacementGenerator(
            100, 100, padding_mask, margin=margin
        )

        # Generate multiple candidates
        random.seed(42)
        for _ in range(50):
            candidate = generator.generate_candidate(20, 20)
            if candidate is not None:
                # Should be within valid bounds with margin
                assert candidate.top >= min_non_padded_top + margin
                assert candidate.left >= min_non_padded_left + margin
                assert candidate.bottom <= max_non_padded - margin
                assert candidate.right <= max_non_padded - margin

    def test_no_valid_region_returns_none(self) -> None:
        """Test behavior when no valid placement region exists."""
        # Fully padded mask
        padding_mask = torch.ones(50, 50)

        generator = PaddingAwarePlacementGenerator(50, 50, padding_mask, margin=5)

        candidate = generator.generate_candidate(20, 20)
        assert candidate is None

    def test_object_too_large_returns_none(self) -> None:
        """Test behavior when object is too large for valid region."""
        padding_mask = torch.ones(100, 100)
        padding_mask[40:60, 40:60] = 0  # Only 20x20 valid region

        generator = PaddingAwarePlacementGenerator(100, 100, padding_mask, margin=5)

        # Object larger than valid region
        candidate = generator.generate_candidate(30, 30)
        assert candidate is None
