"""Tests for COCO dataset integration functionality."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
from PIL import Image
from torchvision import tv_tensors
from torchvision.transforms import v2

from segpaste.integrations.coco import (
    CocoDetectionV2,
    create_coco_dataloader,
    labels_getter,
    segmentation_to_mask,
)
from tests.shared import generate_scale_jitter_transform_strategy


class TestSegmentationToMask:
    """Test cases for segmentation_to_mask function."""

    def test_segmentation_to_mask_rle_dict(self) -> None:
        """Test conversion from RLE dict format."""
        # Create a simple RLE segmentation in uncompressed format
        # This will be processed by frPyObjects and then decoded
        segmentation = {
            "size": [10, 10],
            "counts": [50, 4, 46],  # RLE: 50 zeros, 4 ones, 46 zeros
        }
        canvas_size = (10, 10)

        result = segmentation_to_mask(segmentation, canvas_size)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (10, 10)
        # Should have some non-zero values (the mask area)
        assert result.sum() == 4

    def test_segmentation_to_mask_polygon_list(self) -> None:
        """Test conversion from polygon list format."""
        # Polygon format: list of [x1, y1, x2, y2, ...]
        # Simple rectangle polygon
        segmentation = [[10.0, 10.0, 40.0, 10.0, 40.0, 40.0, 10.0, 40.0]]
        canvas_size = (75, 50)

        result = segmentation_to_mask(segmentation, canvas_size)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (75, 50)
        # Should have some non-zero values (the polygon area)
        assert result.sum() == 900

    def test_segmentation_to_mask_invalid_format(self) -> None:
        """Test error handling for invalid segmentation format."""
        invalid_segmentation = "not_a_dict_or_list"
        canvas_size = (10, 10)

        expected_error = "COCO segmentation expected to be dict or list"
        with pytest.raises(ValueError, match=expected_error):
            segmentation_to_mask(invalid_segmentation, canvas_size)  # pyright: ignore[reportArgumentType]


class TestCocoDetectionV2:
    """Test cases for CocoDetectionV2 dataset class."""

    @pytest.fixture
    def temp_coco_data(self, tmp_path: Path) -> tuple[Path, Path]:
        """Create temporary COCO dataset files for testing."""

        # Create dummy images
        image_dir = tmp_path / "images"
        image_dir.mkdir(parents=True)

        # Create test images
        for i in range(3):
            img = Image.new("RGB", (100, 100), color=(255, 0, 0))
            img.save(image_dir / f"image_{i:03d}.jpg")

        # Create COCO annotations
        annotations = {
            "images": [
                {"id": 1, "file_name": "image_000.jpg", "height": 100, "width": 100},
                {"id": 2, "file_name": "image_001.jpg", "height": 150, "width": 100},
                {"id": 3, "file_name": "image_002.jpg", "height": 100, "width": 150},
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [10, 10, 30, 30],  # XYWH format
                    "area": 900,
                    "segmentation": [[10, 10, 40, 10, 40, 40, 10, 40]],  # Polygon
                    "iscrowd": 0,
                },
                {
                    "id": 2,
                    "image_id": 1,
                    "category_id": 2,
                    "bbox": [50, 50, 20, 20],  # XYWH format
                    "area": 400,
                    "segmentation": [[50, 50, 70, 50, 70, 70, 50, 70]],  # Polygon
                    "iscrowd": 0,
                },
                {
                    "id": 3,
                    "image_id": 2,
                    "category_id": 1,
                    "bbox": [5, 5, 25, 25],  # XYWH format
                    "area": 625,
                    "segmentation": [[5, 5, 30, 5, 30, 30, 5, 30]],  # Polygon
                    "iscrowd": 0,
                },
                # Image 3 has no annotations (will be filtered out)
            ],
            "categories": [
                {"id": 1, "name": "person"},
                {"id": 2, "name": "car"},
            ],
        }

        # Save annotations
        ann_path = tmp_path / "annotations.json"
        with open(ann_path, "w") as f:
            json.dump(annotations, f)

        return image_dir, ann_path

    def test_init_basic(self, temp_coco_data: tuple[str, str]) -> None:
        """Test basic initialization of CocoDetectionV2."""
        image_dir, ann_path = temp_coco_data

        dataset = CocoDetectionV2(image_folder=image_dir, label_path=ann_path)

        assert len(dataset) == 2  # Only images with annotations
        assert dataset.valid_img_ids == [1, 2]  # Sorted IDs with annotations
        expected_keys = ["image_id", "padding_mask", "boxes", "labels", "masks"]
        assert dataset.target_keys == expected_keys

    def test_init_custom_target_keys(self, temp_coco_data: tuple[str, str]) -> None:
        """Test initialization with custom target keys."""
        image_dir, ann_path = temp_coco_data
        custom_keys = ["image_id", "boxes", "labels"]

        dataset = CocoDetectionV2(
            image_folder=image_dir, label_path=ann_path, target_keys=custom_keys
        )

        assert dataset.target_keys == custom_keys

    def test_len(self, temp_coco_data: tuple[str, str]) -> None:
        """Test dataset length."""
        image_dir, ann_path = temp_coco_data
        dataset = CocoDetectionV2(image_dir, ann_path)

        assert len(dataset) == 2

    def test_load_image(self, temp_coco_data: tuple[str, str]) -> None:
        """Test image loading."""
        image_dir, ann_path = temp_coco_data
        dataset = CocoDetectionV2(image_dir, ann_path)

        # Test loading first image
        image = dataset._load_image(1)

        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 100, 100)

    def test_load_target(self, temp_coco_data: tuple[str, str]) -> None:
        """Test target loading."""
        image_dir, ann_path = temp_coco_data
        dataset = CocoDetectionV2(image_dir, ann_path)

        # Test loading targets for first image (has 2 annotations)
        targets = dataset._load_target(1)

        assert isinstance(targets, list)
        assert len(targets) == 2
        assert all(isinstance(t, dict) for t in targets)
        assert targets[0]["category_id"] == 1
        assert targets[1]["category_id"] == 2

    @patch("segpaste.integrations.coco.segmentation_to_mask")
    def test_getitem_with_masks(
        self, mock_seg_to_mask: Any, temp_coco_data: tuple[str, str]
    ) -> None:
        """Test __getitem__ with masks enabled."""
        image_dir, ann_path = temp_coco_data

        # Mock segmentation_to_mask to return dummy masks
        def mock_seg_fn(
            segmentation: Any,  # noqa: ARG001
            canvas_size: tuple[int, int],
        ) -> torch.Tensor:
            return torch.ones(canvas_size[0], canvas_size[1], dtype=torch.uint8)

        mock_seg_to_mask.side_effect = mock_seg_fn

        dataset = CocoDetectionV2(
            image_folder=image_dir,
            label_path=ann_path,
            target_keys=["image_id", "boxes", "labels", "masks"],
        )

        image, target = dataset[0]  # First image (id=1, has 2 annotations)

        # Check image
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 100, 100)

        # Check target structure
        assert isinstance(target, dict)
        assert "image_id" in target
        assert "boxes" in target
        assert "labels" in target
        assert "masks" in target

        # Check target contents
        assert target["image_id"] == 1
        assert isinstance(target["boxes"], tv_tensors.BoundingBoxes)
        assert isinstance(target["labels"], torch.Tensor)
        assert isinstance(target["masks"], tv_tensors.Mask)

        # Check shapes
        assert target["boxes"].shape == (2, 4)  # 2 objects, 4 coords (XYXY)
        assert target["labels"].shape == (2,)  # 2 objects
        assert target["masks"].shape == (2, 100, 100)  # 2 objects, H, W

        # Check labels
        assert target["labels"].tolist() == [1, 2]

        # Check that segmentation_to_mask was called for each annotation
        assert mock_seg_to_mask.call_count == 2

    def test_getitem_without_masks(self, temp_coco_data: tuple[str, str]) -> None:
        """Test __getitem__ without masks."""
        image_dir, ann_path = temp_coco_data

        dataset = CocoDetectionV2(
            image_folder=image_dir,
            label_path=ann_path,
            target_keys=["image_id", "boxes", "labels"],  # No masks
        )

        _, target = dataset[0]

        # Check that masks is not in target
        assert "masks" not in target
        assert "image_id" in target
        assert "boxes" in target
        assert "labels" in target

    def test_getitem_with_padding_mask(self, temp_coco_data: tuple[str, str]) -> None:
        """Test __getitem__ with padding_mask enabled."""
        image_dir, ann_path = temp_coco_data

        dataset = CocoDetectionV2(
            image_folder=image_dir,
            label_path=ann_path,
            target_keys=["image_id", "boxes", "labels", "padding_mask"],
        )

        _, target = dataset[0]

        # Check padding_mask
        assert "padding_mask" in target
        assert isinstance(target["padding_mask"], tv_tensors.Mask)
        assert target["padding_mask"].shape == (1, 100, 100)  # 1xHxW
        assert target["padding_mask"].dtype == torch.bool
        assert not torch.any(target["padding_mask"])

    def test_getitem_empty_target(self, temp_coco_data: tuple[str, str]) -> None:
        """Test __getitem__ behavior with empty targets."""
        image_dir, ann_path = temp_coco_data

        # Create dataset but manually modify valid_img_ids to include
        # image with no annotations
        dataset = CocoDetectionV2(image_dir, ann_path)
        dataset.valid_img_ids = [3]  # Image 3 has no annotations

        _, target = dataset[0]

        # Check empty target structure
        assert target["image_id"] == 3
        assert isinstance(target["boxes"], tv_tensors.BoundingBoxes)
        assert isinstance(target["labels"], torch.Tensor)
        assert isinstance(target["masks"], tv_tensors.Mask)

        # Check empty shapes
        assert target["boxes"].shape == (0, 4)
        assert target["labels"].shape == (0,)
        assert target["masks"].shape == (0, 100, 100)

    def test_getitem_with_transforms(self, temp_coco_data: tuple[str, str]) -> None:
        """Test __getitem__ with transforms applied."""
        image_dir, ann_path = temp_coco_data

        # Define simple transform
        transforms = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize((50, 50)),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )

        dataset = CocoDetectionV2(
            image_folder=image_dir,
            label_path=ann_path,
            transforms=transforms,
            target_keys=["image_id", "boxes", "labels"],
        )

        image, target = dataset[0]

        # Check that image is transformed
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 50, 50)  # CHW format, resized
        assert image.dtype == torch.float32

        # Check that bounding boxes are transformed
        assert isinstance(target["boxes"], tv_tensors.BoundingBoxes)

    @pytest.mark.parametrize(
        "scale,expected_values",
        [
            (0.5, torch.tensor([0, 1])),  # When downscaling, padding is used
            (2.0, torch.tensor([0])),
        ],
    )
    def test_getitem_pixel_masks(
        self,
        temp_coco_data: tuple[str, str],
        scale: float,
        expected_values: torch.Tensor,
    ) -> None:
        """Test __getitem__ with pixel-level masks."""
        image_dir, ann_path = temp_coco_data

        transforms = generate_scale_jitter_transform_strategy(
            min_scale=scale, max_scale=scale
        )

        dataset = CocoDetectionV2(
            image_folder=image_dir,
            label_path=ann_path,
            target_keys=["image_id", "boxes", "labels", "masks", "padding_mask"],
            transforms=transforms,
        )
        image, target = dataset[0]  # First image has 2 annotations
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 256, 256)  # Resized by transform
        assert image.dtype == torch.float32
        assert isinstance(target, dict)
        assert "image_id" in target
        assert "boxes" in target
        assert "labels" in target
        assert "masks" in target
        assert "padding_mask" in target
        assert target["padding_mask"].shape == (1, 256, 256)
        assert torch.equal(target["padding_mask"].unique(), expected_values)

    def test_bbox_format_conversion(self, temp_coco_data: tuple[str, str]) -> None:
        """Test that bounding boxes are correctly converted from XYWH to XYXY."""
        image_dir, ann_path = temp_coco_data

        dataset = CocoDetectionV2(
            image_folder=image_dir, label_path=ann_path, target_keys=["boxes"]
        )

        _, target = dataset[0]  # First image has bbox [10, 10, 30, 30] in XYWH

        boxes = target["boxes"]
        assert isinstance(boxes, tv_tensors.BoundingBoxes)

        # First box should be converted from [10, 10, 30, 30] XYWH to [10, 10, 40, 40]
        # Second box should be converted from [50, 50, 20, 20] XYWH to [50, 50, 70, 70]
        expected_boxes = torch.tensor(
            [[10, 10, 40, 40], [50, 50, 70, 70]], dtype=torch.float32
        )

        torch.testing.assert_close(boxes.data, expected_boxes)
        assert boxes.format == tv_tensors.BoundingBoxFormat.XYXY


class TestLabelsGetter:
    """Test cases for labels_getter function."""

    def test_labels_getter_basic(self) -> None:
        """Test basic functionality of labels_getter."""
        # Create sample data
        image = tv_tensors.Image(torch.rand(3, 100, 100))
        boxes = tv_tensors.BoundingBoxes(
            torch.tensor([[10, 10, 20, 20]]),
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=(100, 100),
        )
        masks = tv_tensors.Mask(torch.ones(1, 100, 100))
        labels = torch.tensor([1])

        target = {"boxes": boxes, "masks": masks, "labels": labels}

        sample = (image, target)

        # Test labels_getter
        result_boxes, result_masks, result_labels = labels_getter(sample)

        assert torch.equal(result_boxes.data, boxes.data)
        assert torch.equal(result_masks.data, masks.data)
        assert torch.equal(result_labels, labels)


class TestCreateCocoDataloader:
    """Test cases for create_coco_dataloader function."""

    @pytest.fixture
    def temp_coco_data(self) -> tuple[str, str]:
        """Create temporary COCO dataset files for testing."""
        temp_dir = tempfile.mkdtemp()

        # Create dummy images
        image_dir = os.path.join(temp_dir, "images")
        os.makedirs(image_dir)

        # Create test images
        for i in range(3):
            img = Image.new("RGB", (100, 100), color=(255, 0, 0))
            img.save(os.path.join(image_dir, f"image_{i:03d}.jpg"))

        # Create COCO annotations
        annotations = {
            "images": [
                {"id": 1, "file_name": "image_000.jpg", "height": 100, "width": 100},
                {"id": 2, "file_name": "image_001.jpg", "height": 100, "width": 100},
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [10, 10, 30, 30],
                    "area": 900,
                    "segmentation": [[10, 10, 40, 10, 40, 40, 10, 40]],
                    "iscrowd": 0,
                },
                {
                    "id": 2,
                    "image_id": 2,
                    "category_id": 2,
                    "bbox": [5, 5, 25, 25],
                    "area": 625,
                    "segmentation": [[5, 5, 30, 5, 30, 30, 5, 30]],
                    "iscrowd": 0,
                },
            ],
            "categories": [
                {"id": 1, "name": "person"},
                {"id": 2, "name": "car"},
            ],
        }

        # Save annotations
        ann_path = os.path.join(temp_dir, "annotations.json")
        with open(ann_path, "w") as f:
            json.dump(annotations, f)

        return image_dir, ann_path

    def test_create_coco_dataloader_basic(
        self, temp_coco_data: tuple[str, str]
    ) -> None:
        """Test basic dataloader creation."""
        image_dir, ann_path = temp_coco_data

        transforms = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )

        dataloader = create_coco_dataloader(
            image_folder=image_dir,
            label_path=ann_path,
            transforms=transforms,
            batch_size=2,
        )

        # Test dataloader properties
        assert dataloader.batch_size == 2
        # Note: shuffle property is not directly accessible on DataLoader

        # Test getting a batch
        batch = next(iter(dataloader))

        assert isinstance(batch, list)
        assert len(batch) == 2  # batch_size

        # Check each sample in batch
        for sample in batch:
            assert isinstance(sample, dict)
            assert "image" in sample
            assert "image_id" in sample
            assert "boxes" in sample
            assert "labels" in sample
            assert "masks" in sample

            # Check tensor types
            assert isinstance(sample["image"], torch.Tensor)
            assert isinstance(sample["boxes"], torch.Tensor)
            assert isinstance(sample["labels"], torch.Tensor)
            assert isinstance(sample["masks"], torch.Tensor)

    def test_coco_collate_fn(self, temp_coco_data: tuple[str, str]) -> None:
        """Test the custom collate function."""
        image_dir, ann_path = temp_coco_data

        transforms = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )

        dataloader = create_coco_dataloader(
            image_folder=image_dir,
            label_path=ann_path,
            transforms=transforms,
            batch_size=1,
        )

        # Get the collate function
        collate_fn = dataloader.collate_fn

        # Create mock batch data
        image = tv_tensors.Image(torch.rand(3, 100, 100))
        target = {
            "image_id": 1,
            "boxes": torch.tensor([[10, 10, 20, 20]]),
            "labels": torch.tensor([1]),
            "masks": torch.ones(1, 100, 100),
        }

        batch = [(image, target)]

        # Test collate function
        result = collate_fn(batch)

        assert isinstance(result, list)
        assert len(result) == 1

        sample = result[0]
        assert "image" in sample
        assert "image_id" in sample
        assert "boxes" in sample
        assert "labels" in sample
        assert "masks" in sample
