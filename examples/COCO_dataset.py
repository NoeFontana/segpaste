"""Download COCO dataset and display sample images."""

import argparse
import logging

import faster_coco_eval as _faster_coco_eval
import fiftyone as fo
import fiftyone.zoo as foz

_faster_coco_eval.init_as_pycocotools()


def download_coco_dataset() -> fo.Dataset:
    """Download all COCO dataset splits using FiftyOne."""
    logging.getLogger().info("Downloading COCO dataset...")

    # Download all splits of COCO dataset
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        splits=["validation"],
        persistent=True,
        overwrite=True,
        drop_existing_dataset=True,
        label_types=["detections", "segmentations"],
    )

    logging.getLogger().info(f"Dataset loaded with {len(dataset)} samples")
    logging.getLogger().info(f"Dataset info: {dataset.info}")

    return dataset


def load_existing_dataset() -> fo.Dataset:
    """Load existing COCO dataset if already downloaded."""
    try:
        # Try to load existing dataset and ensure it has segmentations
        dataset = fo.load_dataset("coco-2017-validation", reload=True)

        # Check if dataset has segmentations, if not reload with segmentations
        logging.getLogger().info(f"Loaded existing dataset with {len(dataset)} samples")  # pyright: ignore[reportArgumentType]
        return dataset
    except ValueError:
        logging.getLogger().info("Dataset not found, downloading...")
        return download_coco_dataset()


def visualize_coco_samples(num_samples: int = 50) -> None:
    """Load COCO dataset and visualize sample images using FiftyOne."""
    dataset = load_existing_dataset()

    # Get samples from validation split
    val_view = dataset.match_tags("validation").limit(num_samples)

    logging.getLogger().info(
        f"Visualizing {len(val_view)} samples from COCO validation set..."
    )
    # Launch FiftyOne App to visualize the samples
    session = fo.launch_app(val_view)

    # Keep the session open
    session.wait(-1)


def create_copy_paste_augmentation(num_samples: int = 20) -> None:
    """Create and visualize copy-paste augmented samples."""
    dataset = load_existing_dataset()

    # Get validation samples
    val_samples = dataset.match_tags("validation").limit(num_samples * 2)

    # Create a new dataset for augmented samples
    augmented_dataset = fo.Dataset("coco-copy-paste-augmented")

    samples_list = list(val_samples)

    for i in range(0, len(samples_list) - 1, 2):
        source_sample = samples_list[i]
        target_sample = samples_list[i + 1]

        # Create augmented sample (simplified version - just metadata for now)
        augmented_sample = fo.Sample(
            filepath=source_sample.filepath,
            tags=["copy-paste-augmented"],
        )

        # Copy ground truth from source
        if source_sample.ground_truth:
            augmented_sample["ground_truth"] = source_sample.ground_truth.copy()

        # Add some detections from target (simplified simulation)
        if (
            target_sample.ground_truth
            and len(target_sample.ground_truth.detections) > 0
        ):
            # Take first detection from target and modify its position
            target_detection = target_sample.ground_truth.detections[0].copy()
            # Modify position to avoid overlap (simplified)
            bbox = target_detection.bounding_box
            target_detection.bounding_box = [
                min(0.7, bbox[0] + 0.3),  # x
                min(0.7, bbox[1] + 0.3),  # y
                min(0.3, bbox[2]),  # width
                min(0.3, bbox[3]),  # height
            ]

            # Add to augmented sample
            if augmented_sample.ground_truth:
                augmented_sample.ground_truth.detections.append(target_detection)

        augmented_dataset.add_sample(augmented_sample)

    logging.getLogger().info(
        f"Created {len(augmented_dataset)} copy-paste augmented samples"
    )

    # Launch FiftyOne App to visualize the augmented samples
    session = fo.launch_app(augmented_dataset)
    session.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description="COCO Dataset Utilities")
    parser.add_argument(
        "mode",
        choices=["download", "visualize", "augment"],
        help="Mode to run: download dataset, visualize samples, or show copy-paste augmentation",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of samples to visualize or augment (default: 50)",
    )

    args = parser.parse_args()

    if args.mode == "download":
        download_coco_dataset()
        logging.getLogger().info("Dataset download completed!")

    elif args.mode == "visualize":
        visualize_coco_samples(args.num_samples)

    elif args.mode == "augment":
        create_copy_paste_augmentation(args.num_samples)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
