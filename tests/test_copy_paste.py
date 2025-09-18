"""Tests for copy-paste augmentation functionality."""

import itertools
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import torch
import torchvision

from segpaste.copy_paste import CopyPasteAugmentation
from segpaste.data_types import CopyPasteConfig, DetectionTarget
from segpaste.dataset import create_coco_dataset
from segpaste.transforms import CopyPasteCollator
from segpaste.utils import boxes_to_masks


def create_sample_detection_target(
    num_objects: int = 2, image_size: Tuple[int, int, int] = (3, 224, 224)
) -> DetectionTarget:
    """Create a sample detection target for testing."""
    c, h, w = image_size

    # Create dummy image
    image = torch.rand(c, h, w)

    # Create dummy boxes
    boxes = torch.tensor([[10, 10, 50, 50], [100, 100, 140, 140]])[:num_objects]

    # Create dummy labels
    labels = torch.tensor([1, 2])[:num_objects]

    # Create dummy masks
    masks = torch.zeros(num_objects, h, w)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.int()
        masks[i, y1:y2, x1:x2] = 1.0

    return DetectionTarget(image=image, boxes=boxes, labels=labels, masks=masks)


class TestCopyPasteAugmentation:
    """Test cases for CopyPasteAugmentation."""

    def test_copy_paste_with_probability_0(self) -> None:
        """Test that no augmentation is applied when probability is 0."""
        config = CopyPasteConfig(paste_probability=0.0)
        augmentation = CopyPasteAugmentation(config)

        target_data = create_sample_detection_target()
        source_objects = [create_sample_detection_target(num_objects=1)]

        result = augmentation.transform(target_data, source_objects)

        # Should return original data unchanged
        assert torch.equal(result.image, target_data.image)
        assert torch.equal(result.boxes, target_data.boxes)
        assert torch.equal(result.labels, target_data.labels)
        assert torch.equal(result.masks, target_data.masks)

    def test_copy_paste_with_probability_1(self) -> None:
        """Test that augmentation is applied when probability is 1."""
        config = CopyPasteConfig(
            paste_probability=1.0, max_paste_objects=1, min_paste_objects=1
        )
        augmentation = CopyPasteAugmentation(config)

        target_data = create_sample_detection_target()
        source_objects = [create_sample_detection_target(num_objects=1)]

        result = augmentation.transform(target_data, source_objects)

        # Should have more objects than original
        assert result.boxes.shape[0] >= target_data.boxes.shape[0]
        assert result.labels.shape[0] >= target_data.labels.shape[0]
        assert result.masks.shape[0] >= target_data.masks.shape[0]

    def test_copy_paste_empty_source_objects(self) -> None:
        """Test behavior with empty source objects."""
        augmentation = CopyPasteAugmentation(CopyPasteConfig())

        target_data = create_sample_detection_target()
        source_objects: list[DetectionTarget] = []

        result = augmentation.transform(target_data, source_objects)

        # Should return original data unchanged
        assert torch.equal(result.image, target_data.image)
        assert torch.equal(result.boxes, target_data.boxes)
        assert torch.equal(result.labels, target_data.labels)
        assert torch.equal(result.masks, target_data.masks)


class TestUtils:
    """Test cases for utility functions."""

    def test_boxes_to_masks(self) -> None:
        """Test conversion from boxes to masks."""
        boxes = torch.tensor([[10, 10, 20, 20], [30, 30, 40, 40]])

        masks = boxes_to_masks(boxes, height=50, width=50)

        assert masks.shape == (2, 50, 50)
        assert masks[0, 10:20, 10:20].sum() == 100  # 10x10 box
        assert masks[1, 30:40, 30:40].sum() == 100  # 10x10 box


class TestDetectionTarget:
    """Test cases for DetectionTarget data structure."""

    def test_detection_target_creation(self) -> None:
        """Test DetectionTarget creation and access."""
        image = torch.rand(3, 224, 224)
        boxes = torch.tensor([[10, 10, 50, 50]])
        labels = torch.tensor([1])
        masks = torch.zeros(1, 224, 224)

        target = DetectionTarget(image=image, boxes=boxes, labels=labels, masks=masks)

        assert torch.equal(target.image, image)
        assert torch.equal(target.boxes, boxes)
        assert torch.equal(target.labels, labels)
        assert torch.equal(target.masks, masks)


class TestCopyPasteCollator:
    """Test cases for CopyPasteCollator."""

    def test_collator_init(self) -> None:
        """Test CopyPasteCollator initialization."""
        config = CopyPasteConfig(paste_probability=0.8)
        collator = CopyPasteCollator(config)

        assert collator.copy_paste.config.paste_probability == 0.8

    def test_collator_empty_batch(self) -> None:
        """Test collator with empty batch."""
        config = CopyPasteConfig()
        collator = CopyPasteCollator(config)

        result = collator([])
        assert result == {}

    def test_collator_with_coco_dataset(self) -> None:
        """Test CopyPasteCollator with real COCO dataset."""
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)

        # Default COCO dataset path
        default_path: Path = Path.home() / "fiftyone" / "coco-2017" / "validation"
        dataset_path: str = os.environ.get("COCO_DATASET_PATH", str(default_path))

        # Check if dataset exists
        val_images_path: str = os.path.join(dataset_path, "data")
        annotations_path: str = os.path.join(dataset_path, "labels.json")

        if not (os.path.exists(val_images_path) and os.path.exists(annotations_path)):
            pytest.skip(f"COCO dataset not found at {dataset_path}")

        dataloader: torch.utils.data.DataLoader = create_coco_dataset(
            image_folder=val_images_path, label_path=annotations_path, batch_size=4
        )

        # Get a batch of samples
        batch_samples: List[Dict[str, torch.Tensor]] = []
        for i, samples in enumerate(itertools.islice(dataloader, 10)):
            if len(samples) != 4:  # Need at least 4 samples for copy-paste
                continue
            batch_samples = samples[:4]  # Take first 4 samples

            # Test collator
            config: CopyPasteConfig = CopyPasteConfig(
                paste_probability=1.0,  # Always apply for testing
                max_paste_objects=20,
                min_paste_objects=5,
                scale_range=(0.5, 2.0),
            )
            collator: CopyPasteCollator = CopyPasteCollator(config)

            # Apply collator to batch
            collated_batch: Dict[str, Any] = collator(batch_samples)

            torchvision.utils.save_image(
                [sample["image"] for sample in batch_samples],
                f"original_images_{i}.png",
                nrow=4,
            )
            torchvision.utils.save_image(
                collated_batch["images"], f"pasted_image_{i}.png", nrow=4
            )

            # Verify collated batch structure
            assert "images" in collated_batch
            assert "boxes" in collated_batch
            assert "labels" in collated_batch
            assert "masks" in collated_batch

            # Check batch dimensions
            assert isinstance(collated_batch["images"], torch.Tensor)
            assert collated_batch["images"].shape[0] <= len(batch_samples)

            # Check that boxes, labels, and masks are lists
            assert isinstance(collated_batch["boxes"], list)
            assert isinstance(collated_batch["labels"], list)
            assert isinstance(collated_batch["masks"], list)

            # Verify each sample in batch
            for i in range(len(collated_batch["boxes"])):
                boxes = collated_batch["boxes"][i]
                labels = collated_batch["labels"][i]
                masks = collated_batch["masks"][i]

                assert isinstance(boxes, torch.Tensor)
                assert isinstance(labels, torch.Tensor)
                assert isinstance(masks, torch.Tensor)

                # Check tensor shapes are consistent
                assert boxes.shape[0] == labels.shape[0] == masks.shape[0]
                assert boxes.shape[1] == 4  # xyxy format
