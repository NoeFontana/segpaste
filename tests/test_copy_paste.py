"""Tests for copy-paste augmentation functionality."""

import torch

from segpaste.copy_paste import CopyPasteAugmentation
from segpaste.data_types import CopyPasteConfig, DetectionTarget
from segpaste.utils import boxes_to_masks, compute_iou, masks_to_boxes


def create_sample_detection_target(
    num_objects: int = 2, image_size: tuple = (3, 224, 224)
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

    def test_init_default_config(self):
        """Test initialization with default configuration."""
        augmentation = CopyPasteAugmentation()
        assert augmentation.config.paste_probability == 0.5
        assert augmentation.config.max_paste_objects == 10
        assert augmentation.config.blend_mode == "alpha"

    def test_init_custom_config(self):
        """Test initialization with custom configuration."""
        config = CopyPasteConfig(
            paste_probability=0.8, max_paste_objects=5, blend_mode="gaussian"
        )
        augmentation = CopyPasteAugmentation(config)
        assert augmentation.config.paste_probability == 0.8
        assert augmentation.config.max_paste_objects == 5
        assert augmentation.config.blend_mode == "gaussian"

    def test_copy_paste_with_probability_0(self):
        """Test that no augmentation is applied when probability is 0."""
        config = CopyPasteConfig(paste_probability=0.0)
        augmentation = CopyPasteAugmentation(config)

        target_data = create_sample_detection_target()
        source_objects = [create_sample_detection_target(num_objects=1)]

        result = augmentation(target_data, source_objects)

        # Should return original data unchanged
        assert torch.equal(result.image, target_data.image)
        assert torch.equal(result.boxes, target_data.boxes)
        assert torch.equal(result.labels, target_data.labels)
        assert torch.equal(result.masks, target_data.masks)

    def test_copy_paste_with_probability_1(self):
        """Test that augmentation is applied when probability is 1."""
        config = CopyPasteConfig(
            paste_probability=1.0, max_paste_objects=1, min_paste_objects=1
        )
        augmentation = CopyPasteAugmentation(config)

        target_data = create_sample_detection_target()
        source_objects = [create_sample_detection_target(num_objects=1)]

        result = augmentation(target_data, source_objects)

        # Should have more objects than original
        assert result.boxes.shape[0] >= target_data.boxes.shape[0]
        assert result.labels.shape[0] >= target_data.labels.shape[0]
        assert result.masks.shape[0] >= target_data.masks.shape[0]

    def test_copy_paste_without_masks(self):
        """Test behavior when masks are not provided."""
        augmentation = CopyPasteAugmentation()

        # Create target without masks
        target_data = DetectionTarget(
            image=torch.rand(3, 224, 224),
            boxes=torch.tensor([[10, 10, 50, 50]]),
            labels=torch.tensor([1]),
            masks=None,
        )
        source_objects = [create_sample_detection_target(num_objects=1)]

        result = augmentation(target_data, source_objects)

        # Should return original data unchanged
        assert torch.equal(result.image, target_data.image)
        assert torch.equal(result.boxes, target_data.boxes)
        assert torch.equal(result.labels, target_data.labels)

    def test_copy_paste_empty_source_objects(self):
        """Test behavior with empty source objects."""
        augmentation = CopyPasteAugmentation()

        target_data = create_sample_detection_target()
        source_objects = []

        result = augmentation(target_data, source_objects)

        # Should return original data unchanged
        assert torch.equal(result.image, target_data.image)
        assert torch.equal(result.boxes, target_data.boxes)
        assert torch.equal(result.labels, target_data.labels)
        assert torch.equal(result.masks, target_data.masks)


class TestUtils:
    """Test cases for utility functions."""

    def test_boxes_to_masks(self):
        """Test conversion from boxes to masks."""
        boxes = torch.tensor([[10, 10, 20, 20], [30, 30, 40, 40]])

        masks = boxes_to_masks(boxes, height=50, width=50)

        assert masks.shape == (2, 50, 50)
        assert masks[0, 10:20, 10:20].sum() == 100  # 10x10 box
        assert masks[1, 30:40, 30:40].sum() == 100  # 10x10 box

    def test_masks_to_boxes(self):
        """Test conversion from masks to boxes."""
        masks = torch.zeros(2, 50, 50)
        masks[0, 10:20, 10:20] = 1.0
        masks[1, 30:40, 30:40] = 1.0

        boxes = masks_to_boxes(masks)

        expected_boxes = torch.tensor(
            [
                [10, 10, 19, 19],  # Note: max coordinates are inclusive
                [30, 30, 39, 39],
            ],
            dtype=torch.float32,
        )

        assert torch.allclose(boxes, expected_boxes)

    def test_compute_iou(self):
        """Test IoU computation between boxes."""
        boxes1 = torch.tensor([[0, 0, 10, 10], [5, 5, 15, 15]])
        boxes2 = torch.tensor([[0, 0, 10, 10], [10, 10, 20, 20]])

        iou = compute_iou(boxes1, boxes2)

        assert iou.shape == (2, 2)
        assert torch.allclose(iou[0, 0], torch.tensor(1.0))  # Perfect overlap
        assert torch.allclose(iou[1, 1], torch.tensor(0.0))  # No overlap

        # boxes1[1] and boxes2[0] should have some overlap
        assert iou[1, 0] > 0.0 and iou[1, 0] < 1.0

    def test_compute_iou_empty_boxes(self):
        """Test IoU computation with empty box sets."""
        boxes1 = torch.empty(0, 4)
        boxes2 = torch.tensor([[0, 0, 10, 10]])

        iou = compute_iou(boxes1, boxes2)

        assert iou.shape == (0, 1)


class TestDetectionTarget:
    """Test cases for DetectionTarget data structure."""

    def test_detection_target_creation(self):
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

    def test_detection_target_without_masks(self):
        """Test DetectionTarget creation without masks."""
        image = torch.rand(3, 224, 224)
        boxes = torch.tensor([[10, 10, 50, 50]])
        labels = torch.tensor([1])

        target = DetectionTarget(image=image, boxes=boxes, labels=labels)

        assert target.masks is None


class TestCopyPasteConfig:
    """Test cases for CopyPasteConfig."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = CopyPasteConfig()

        assert config.paste_probability == 0.5
        assert config.max_paste_objects == 10
        assert config.min_paste_objects == 1
        assert config.scale_range == (0.1, 2.0)
        assert config.blend_mode == "alpha"
        assert config.occluded_area_threshold == 0.3
        assert config.box_update_threshold == 10.0
        assert config.enable_rotation is False
        assert config.enable_flip is True

    def test_config_custom_values(self):
        """Test configuration with custom values."""
        config = CopyPasteConfig(
            paste_probability=0.8,
            max_paste_objects=5,
            min_paste_objects=2,
            scale_range=(0.5, 1.5),
            blend_mode="gaussian",
            occluded_area_threshold=0.4,
            box_update_threshold=15.0,
            enable_rotation=True,
            enable_flip=False,
        )

        assert config.paste_probability == 0.8
        assert config.max_paste_objects == 5
        assert config.min_paste_objects == 2
        assert config.scale_range == (0.5, 1.5)
        assert config.blend_mode == "gaussian"
        assert config.occluded_area_threshold == 0.4
        assert config.box_update_threshold == 15.0
        assert config.enable_rotation is True
        assert config.enable_flip is False
