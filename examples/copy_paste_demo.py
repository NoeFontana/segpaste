#!/usr/bin/env python3
"""Demonstration of copy-paste augmentation functionality."""

import argparse
import logging
import os
from pathlib import Path

import torch
from torchvision.transforms import v2

from segpaste.augmentation import CopyPasteAugmentation
from segpaste.config import CopyPasteConfig
from segpaste.integrations import create_coco_dataloader
from segpaste.types import DenseSample


def demonstrate_copy_paste_augmentation() -> None:
    """Demonstrate copy-paste augmentation with COCO dataset."""
    default_path = Path.home() / "fiftyone" / "coco-2017" / "validation"
    dataset_path = os.environ.get("COCO_DATASET_PATH", str(default_path))

    val_images_path = os.path.join(dataset_path, "data")
    annotations_path = os.path.join(dataset_path, "labels.json")

    if not (os.path.exists(val_images_path) and os.path.exists(annotations_path)):
        logging.getLogger().warning(
            f"COCO dataset not found at {dataset_path}. "
            "Please set COCO_DATASET_PATH environment variable."
        )
        return

    transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((256, 256)),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )

    dataloader = create_coco_dataloader(
        image_folder=val_images_path,
        label_path=annotations_path,
        transforms=transforms,
        batch_size=4,
    )

    samples: list[DenseSample] = []
    for i, batch in enumerate(dataloader):
        if i >= 2:
            break
        samples.extend(batch)

    if len(samples) < 4:
        logging.getLogger().error("Need at least 4 samples for demonstration")
        return

    config = CopyPasteConfig(
        paste_probability=1.0,
        max_paste_objects=3,
        min_paste_objects=1,
        scale_range=(0.5, 2.0),
    )

    copy_paste_aug = CopyPasteAugmentation(config)

    logging.getLogger().info(f"Demonstrating copy-paste on {len(samples)} samples...")

    augmented_samples: list[DenseSample] = []
    for i in range(len(samples) - 1):
        target_data = samples[i]
        source_objects = samples[i + 1 : i + 3]
        try:
            augmented_target = copy_paste_aug.transform(target_data, source_objects)
            augmented_samples.append(augmented_target)

            logging.getLogger().info(
                f"Sample {i}: Original had {target_data.boxes.shape[0]} objects, "
                f"augmented has {augmented_target.boxes.shape[0]} objects"
            )

        except Exception as e:
            logging.getLogger().warning(
                f"Failed to apply copy-paste to sample {i}: {e}"
            )
            continue

    logging.getLogger().info(
        f"Successfully created {len(augmented_samples)} augmented samples"
    )

    if os.environ.get("SAVE_DEMO_IMAGES", "0") == "1":
        save_demo_images(samples[: len(augmented_samples)], augmented_samples)


def save_demo_images(
    original_samples: list[DenseSample], augmented_samples: list[DenseSample]
) -> None:
    """Save original and augmented images for visual comparison using torchvision."""
    import torchvision.utils as tv_utils

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    for i, (original, augmented) in enumerate(
        zip(original_samples, augmented_samples, strict=True)
    ):
        orig_img = (original.image.as_subclass(torch.Tensor) * 255).to(torch.uint8)
        aug_img = (augmented.image.as_subclass(torch.Tensor) * 255).to(torch.uint8)

        if len(original.boxes) > 0:
            orig_labels = [str(label.item()) for label in original.labels]
            orig_with_boxes = tv_utils.draw_bounding_boxes(
                orig_img,
                original.boxes.as_subclass(torch.Tensor),
                labels=orig_labels,
                colors="red",
                width=2,
            )
        else:
            orig_with_boxes = orig_img

        if len(augmented.boxes) > 0:
            aug_labels = [str(label.item()) for label in augmented.labels]
            aug_with_boxes = tv_utils.draw_bounding_boxes(
                aug_img,
                augmented.boxes.as_subclass(torch.Tensor),
                labels=aug_labels,
                colors="blue",
                width=2,
            )
        else:
            aug_with_boxes = aug_img

        grid = tv_utils.make_grid(
            [orig_with_boxes, aug_with_boxes],
            nrow=2,
            padding=10,
            pad_value=255,
        )

        output_path = output_dir / f"copy_paste_demo_{i}.png"
        tv_utils.save_image(grid.float() / 255.0, output_path)

    logging.getLogger().info(f"Demo images saved to {output_dir}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Copy-Paste Augmentation Demo")
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save demo images for visual inspection",
    )

    args = parser.parse_args()

    if args.save_images:
        os.environ["SAVE_DEMO_IMAGES"] = "1"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    demonstrate_copy_paste_augmentation()


if __name__ == "__main__":
    main()
