"""Tests for copy-paste augmentation functionality."""

import itertools
import logging
import os
import random
from pathlib import Path

import numpy as np
import pytest
import torch
import torchvision
from torchvision import tv_tensors
from torchvision.transforms import v2

from segpaste.augmentation import (
    CopyPasteAugmentation,
    CopyPasteCollator,
)
from segpaste.config import CopyPasteConfig
from segpaste.integrations import create_coco_dataloader
from segpaste.processing import boxes_to_masks
from segpaste.types import BatchedDenseSample, DenseSample, InstanceMask
from tests.shared import (
    generate_resize_transform_strategy,
    generate_scale_jitter_transform_strategy,
)


def create_sample_dense_sample(
    num_objects: int = 2, image_size: tuple[int, int, int] = (3, 224, 224)
) -> DenseSample:
    """Create a sample DenseSample (INSTANCE modality) for testing."""
    c, h, w = image_size

    image = torch.rand(c, h, w)
    raw_boxes = torch.tensor(
        [[10, 10, 50, 50], [100, 100, 140, 140]], dtype=torch.float32
    )[:num_objects]
    labels = torch.tensor([1, 2], dtype=torch.int64)[:num_objects]

    masks = torch.zeros(num_objects, h, w, dtype=torch.bool)
    for i, box in enumerate(raw_boxes):
        x1, y1, x2, y2 = box.int()
        masks[i, y1:y2, x1:x2] = True

    return DenseSample(
        image=tv_tensors.Image(image),
        boxes=tv_tensors.BoundingBoxes(  # pyright: ignore[reportCallIssue]
            raw_boxes, format=tv_tensors.BoundingBoxFormat.XYXY, canvas_size=(h, w)
        ),
        labels=labels,
        instance_ids=torch.arange(num_objects, dtype=torch.int32),
        instance_masks=InstanceMask(masks),
    )


def _masks_tensor(sample: DenseSample) -> torch.Tensor:
    assert sample.instance_masks is not None
    return sample.instance_masks.as_subclass(torch.Tensor)


class TestCopyPasteAugmentation:
    """Test cases for CopyPasteAugmentation."""

    def test_copy_paste_with_probability_0(self) -> None:
        """No augmentation is applied when probability is 0."""
        config = CopyPasteConfig(paste_probability=0.0)
        augmentation = CopyPasteAugmentation(config)

        target_data = create_sample_dense_sample()
        source_objects = [create_sample_dense_sample(num_objects=1)]

        result = augmentation.transform(target_data, source_objects)

        assert result is target_data

    def test_copy_paste_with_probability_1(self) -> None:
        """Augmentation runs when probability is 1 and never loses objects."""
        config = CopyPasteConfig(
            paste_probability=1.0, max_paste_objects=1, min_paste_objects=1
        )
        augmentation = CopyPasteAugmentation(config)

        target_data = create_sample_dense_sample()
        source_objects = [create_sample_dense_sample(num_objects=1)]

        result = augmentation.transform(target_data, source_objects)
        assert result.instance_masks is not None

        assert result.boxes.shape[0] >= target_data.boxes.shape[0]
        assert result.labels.shape[0] >= target_data.labels.shape[0]
        assert _masks_tensor(result).shape[0] >= _masks_tensor(target_data).shape[0]

    def test_copy_paste_empty_source_objects(self) -> None:
        """Behavior with empty source objects — returns target unchanged."""
        augmentation = CopyPasteAugmentation(CopyPasteConfig())

        target_data = create_sample_dense_sample()
        source_objects: list[DenseSample] = []

        result = augmentation.transform(target_data, source_objects)

        assert result is target_data


class TestUtils:
    """Test cases for utility functions."""

    def test_boxes_to_masks(self) -> None:
        """Test conversion from boxes to masks."""
        boxes = torch.tensor([[10, 10, 20, 20], [30, 30, 40, 40]])

        masks = boxes_to_masks(boxes, height=50, width=50)

        assert masks.shape == (2, 50, 50)
        assert masks[0, 10:20, 10:20].sum() == 100  # 10x10 box
        assert masks[1, 30:40, 30:40].sum() == 100  # 10x10 box


class TestCopyPasteCollator:
    """Test cases for CopyPasteCollator."""

    def test_collator_init(self) -> None:
        """Test CopyPasteCollator initialization."""
        config = CopyPasteConfig(paste_probability=0.8)
        collator = CopyPasteCollator(CopyPasteAugmentation(config))

        assert collator.copy_paste.config.paste_probability == 0.8

    def test_collator_empty_batch(self) -> None:
        """Empty batch yields an empty :class:`BatchedDenseSample`."""
        config = CopyPasteConfig()
        collator = CopyPasteCollator(CopyPasteAugmentation(config))

        result = collator([])
        assert isinstance(result, BatchedDenseSample)
        assert result.batch_size == 0

    def test_collator_small_batch(self) -> None:
        """Two-sample batch — paste runs, BatchedDenseSample is well-formed."""
        config = CopyPasteConfig(
            paste_probability=1.0,
            max_paste_objects=1,
            min_paste_objects=1,
        )
        collator = CopyPasteCollator(CopyPasteAugmentation(config))
        batch = [create_sample_dense_sample(), create_sample_dense_sample()]
        result = collator(batch)

        assert isinstance(result, BatchedDenseSample)
        assert result.batch_size == 2
        assert result.instance_masks is not None
        assert result.instance_ids is not None
        assert len(result.boxes) == 2
        assert len(result.labels) == 2

    @pytest.mark.parametrize(
        "transforms",
        [
            generate_resize_transform_strategy(),
            generate_scale_jitter_transform_strategy(),
        ],
    )
    def test_collator_with_coco_dataset(
        self, transforms: v2.Transform, tmp_path: Path
    ) -> None:
        """Test CopyPasteCollator with real COCO dataset."""
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)

        default_path: Path = Path.home() / "fiftyone" / "coco-2017" / "validation"
        dataset_path: str = os.environ.get("COCO_DATASET_PATH", str(default_path))

        val_images_path: str = os.path.join(dataset_path, "data")
        annotations_path: str = os.path.join(dataset_path, "labels.json")

        if not (os.path.exists(val_images_path) and os.path.exists(annotations_path)):
            pytest.skip(f"COCO dataset not found at {dataset_path}")

        if os.environ.get("SAVE_TEST_IMAGES", "0") == "1":
            logging.getLogger().info(f"Images will be saved to {tmp_path}")

        config: CopyPasteConfig = CopyPasteConfig(
            paste_probability=1.0,
            max_paste_objects=20,
            min_paste_objects=5,
            scale_range=(0.5, 2.0),
        )
        collator: CopyPasteCollator = CopyPasteCollator(CopyPasteAugmentation(config))

        dataloader = create_coco_dataloader(
            image_folder=val_images_path,
            label_path=annotations_path,
            transforms=transforms,
            batch_size=4,
            collate_fn=collator,
        )

        for i, batched in enumerate(itertools.islice(dataloader, 2)):
            if os.environ.get("SAVE_TEST_IMAGES", "0") == "1":
                torchvision.utils.save_image(
                    batched.images.as_subclass(torch.Tensor),
                    f"{tmp_path}/pasted_image_{i}.png",
                    nrow=4,
                )

            assert isinstance(batched, BatchedDenseSample)
            assert batched.batch_size == 4
            assert batched.instance_masks is not None
            assert batched.instance_ids is not None
            for boxes_i, labels_i, masks_i in zip(
                batched.boxes, batched.labels, batched.instance_masks, strict=True
            ):
                assert boxes_i.shape[0] == labels_i.shape[0]
                assert boxes_i.shape[0] == masks_i.shape[0]
                assert boxes_i.shape[1] == 4
