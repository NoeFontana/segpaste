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
from segpaste.types import DenseSample
from tests.shared import generate_scale_jitter_transform_strategy


class TestSegmentationToMask:
    """Test cases for segmentation_to_mask function."""

    def test_segmentation_to_mask_rle_dict(self) -> None:
        """Test conversion from RLE dict format."""
        segmentation = {
            "size": [10, 10],
            "counts": [50, 4, 46],
        }
        canvas_size = (10, 10)

        result = segmentation_to_mask(segmentation, canvas_size)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (10, 10)
        assert result.sum() == 4

    def test_segmentation_to_mask_polygon_list(self) -> None:
        """Test conversion from polygon list format."""
        segmentation = [[10.0, 10.0, 40.0, 10.0, 40.0, 40.0, 10.0, 40.0]]
        canvas_size = (75, 50)

        result = segmentation_to_mask(segmentation, canvas_size)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (75, 50)
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

        image_dir = tmp_path / "images"
        image_dir.mkdir(parents=True)

        for i in range(3):
            img = Image.new("RGB", (100, 100), color=(255, 0, 0))
            img.save(image_dir / f"image_{i:03d}.jpg")

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
                    "bbox": [10, 10, 30, 30],
                    "area": 900,
                    "segmentation": [[10, 10, 40, 10, 40, 40, 10, 40]],
                    "iscrowd": 0,
                },
                {
                    "id": 2,
                    "image_id": 1,
                    "category_id": 2,
                    "bbox": [50, 50, 20, 20],
                    "area": 400,
                    "segmentation": [[50, 50, 70, 50, 70, 70, 50, 70]],
                    "iscrowd": 0,
                },
                {
                    "id": 3,
                    "image_id": 2,
                    "category_id": 1,
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

        ann_path = tmp_path / "annotations.json"
        with open(ann_path, "w") as f:
            json.dump(annotations, f)

        return image_dir, ann_path

    def test_init_basic(self, temp_coco_data: tuple[str, str]) -> None:
        """Test basic initialization of CocoDetectionV2."""
        image_dir, ann_path = temp_coco_data

        dataset = CocoDetectionV2(image_folder=image_dir, label_path=ann_path)

        assert len(dataset) == 2
        assert dataset.valid_img_ids == [1, 2]
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

        image = dataset._load_image(1)  # pyright: ignore[reportPrivateUsage]

        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 100, 100)

    def test_load_target(self, temp_coco_data: tuple[str, str]) -> None:
        """Test target loading."""
        image_dir, ann_path = temp_coco_data
        dataset = CocoDetectionV2(image_dir, ann_path)

        targets = dataset._load_target(1)  # pyright: ignore[reportPrivateUsage]

        assert isinstance(targets, list)
        assert len(targets) == 2
        assert all(isinstance(t, dict) for t in targets)
        assert targets[0]["category_id"] == 1
        assert targets[1]["category_id"] == 2

    @patch("segpaste.integrations.coco.segmentation_to_mask")
    def test_getitem_with_masks(
        self, mock_seg_to_mask: Any, temp_coco_data: tuple[str, str]
    ) -> None:
        """``__getitem__`` returns a populated :class:`DenseSample`."""
        image_dir, ann_path = temp_coco_data

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

        sample = dataset[0]

        assert isinstance(sample, DenseSample)
        assert sample.image.shape == (3, 100, 100)
        assert isinstance(sample.boxes, tv_tensors.BoundingBoxes)
        assert sample.boxes.shape == (2, 4)
        assert sample.labels.shape == (2,)
        assert sample.labels.tolist() == [1, 2]

        assert sample.instance_masks is not None
        assert sample.instance_masks.shape == (2, 100, 100)
        assert sample.instance_masks.dtype == torch.bool

        assert sample.instance_ids is not None
        assert sample.instance_ids.dtype == torch.int32
        assert sample.instance_ids.tolist() == [0, 1]

        assert mock_seg_to_mask.call_count == 2

    def test_getitem_without_masks(self, temp_coco_data: tuple[str, str]) -> None:
        """Test __getitem__ without masks — instance_masks/ids stay None."""
        image_dir, ann_path = temp_coco_data

        dataset = CocoDetectionV2(
            image_folder=image_dir,
            label_path=ann_path,
            target_keys=["image_id", "boxes", "labels"],
        )

        sample = dataset[0]

        assert isinstance(sample, DenseSample)
        assert sample.instance_masks is None
        assert sample.instance_ids is None
        assert sample.boxes.shape[0] == 2
        assert sample.labels.shape[0] == 2

    def test_getitem_with_padding_mask(self, temp_coco_data: tuple[str, str]) -> None:
        """Test __getitem__ with padding_mask enabled."""
        image_dir, ann_path = temp_coco_data

        dataset = CocoDetectionV2(
            image_folder=image_dir,
            label_path=ann_path,
            target_keys=["image_id", "boxes", "labels", "padding_mask"],
        )

        sample = dataset[0]

        assert sample.padding_mask is not None
        assert sample.padding_mask.shape == (1, 100, 100)
        assert sample.padding_mask.dtype == torch.bool
        assert not torch.any(sample.padding_mask)

    def test_getitem_empty_target(self, temp_coco_data: tuple[str, str]) -> None:
        """Test __getitem__ behavior with empty targets."""
        image_dir, ann_path = temp_coco_data

        dataset = CocoDetectionV2(image_dir, ann_path)
        dataset.valid_img_ids = [3]

        sample = dataset[0]

        assert isinstance(sample, DenseSample)
        assert sample.boxes.shape == (0, 4)
        assert sample.labels.shape == (0,)
        assert sample.instance_masks is not None
        assert sample.instance_masks.shape == (0, 100, 100)
        assert sample.instance_ids is not None
        assert sample.instance_ids.shape == (0,)

    def test_getitem_with_transforms(self, temp_coco_data: tuple[str, str]) -> None:
        """Test __getitem__ with transforms applied."""
        image_dir, ann_path = temp_coco_data

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

        sample = dataset[0]

        assert sample.image.shape == (3, 50, 50)
        assert sample.image.dtype == torch.float32
        assert isinstance(sample.boxes, tv_tensors.BoundingBoxes)

    @pytest.mark.parametrize(
        "scale,expected_values",
        [
            (0.5, torch.tensor([0, 1])),
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
        sample = dataset[0]

        assert sample.image.shape == (3, 256, 256)
        assert sample.image.dtype == torch.float32
        assert sample.instance_masks is not None
        assert sample.padding_mask is not None
        assert sample.padding_mask.shape == (1, 256, 256)
        assert torch.equal(sample.padding_mask.unique(), expected_values)

    def test_bbox_format_conversion(self, temp_coco_data: tuple[str, str]) -> None:
        """Test that bounding boxes are correctly converted from XYWH to XYXY."""
        image_dir, ann_path = temp_coco_data

        dataset = CocoDetectionV2(
            image_folder=image_dir, label_path=ann_path, target_keys=["boxes", "labels"]
        )

        sample = dataset[0]

        boxes = sample.boxes
        assert isinstance(boxes, tv_tensors.BoundingBoxes)

        expected_boxes = torch.tensor(
            [[10, 10, 40, 40], [50, 50, 70, 70]], dtype=torch.float32
        )

        torch.testing.assert_close(boxes.data, expected_boxes)
        assert boxes.format == tv_tensors.BoundingBoxFormat.XYXY


class TestLabelsGetter:
    """Test cases for labels_getter function."""

    def test_labels_getter_basic(self) -> None:
        """Test basic functionality of labels_getter."""
        image = tv_tensors.Image(torch.rand(3, 100, 100))
        boxes = tv_tensors.BoundingBoxes(  # pyright: ignore[reportCallIssue]
            torch.tensor([[10, 10, 20, 20]]),
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=(100, 100),
        )
        masks = tv_tensors.Mask(torch.ones(1, 100, 100))
        labels = torch.tensor([1])

        target = {"boxes": boxes, "masks": masks, "labels": labels}

        sample = (image, target)

        result_boxes, result_masks, result_labels = labels_getter(sample)  # pyright: ignore[reportArgumentType]

        assert torch.equal(result_boxes.data, boxes.data)
        assert torch.equal(result_masks.data, masks.data)
        assert torch.equal(result_labels, labels)


class TestCreateCocoDataloader:
    """Test cases for create_coco_dataloader function."""

    @pytest.fixture
    def temp_coco_data(self) -> tuple[str, str]:
        """Create temporary COCO dataset files for testing."""
        temp_dir = tempfile.mkdtemp()

        image_dir = os.path.join(temp_dir, "images")
        os.makedirs(image_dir)

        for i in range(3):
            img = Image.new("RGB", (100, 100), color=(255, 0, 0))
            img.save(os.path.join(image_dir, f"image_{i:03d}.jpg"))

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

        ann_path = os.path.join(temp_dir, "annotations.json")
        with open(ann_path, "w") as f:
            json.dump(annotations, f)

        return image_dir, ann_path

    def test_create_coco_dataloader_basic(
        self, temp_coco_data: tuple[str, str]
    ) -> None:
        """Default dataloader yields ``list[DenseSample]``."""
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

        assert dataloader.batch_size == 2

        batch = next(iter(dataloader))

        assert isinstance(batch, list)
        assert len(batch) == 2

        for sample in batch:
            assert isinstance(sample, DenseSample)
            assert sample.instance_masks is not None
            assert sample.instance_ids is not None
            assert sample.boxes.shape[0] == sample.labels.shape[0]
            assert sample.instance_masks.shape[0] == sample.boxes.shape[0]
