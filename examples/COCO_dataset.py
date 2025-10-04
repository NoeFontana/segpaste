"""Download COCO dataset and display sample images."""

import argparse
import logging

import fiftyone as fo
import fiftyone.zoo as foz
import numpy as np
import torch
from PIL import Image

from segpaste.augmentation import CopyPasteAugmentation
from segpaste.config import CopyPasteConfig
from segpaste.types import DetectionTarget


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


def fiftyone_to_detection_target(sample: fo.Sample) -> DetectionTarget:
    """Convert FiftyOne sample to DetectionTarget format."""
    # Load image
    image_pil = Image.open(sample.filepath).convert("RGB")
    image_np = np.array(image_pil)
    # Convert to tensor [C, H, W] format
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0

    # Use segmentations field which contains masks
    if (
        not hasattr(sample, "segmentations")
        or not sample.segmentations
        or len(sample.segmentations.detections) == 0
    ):
        # Empty detection target
        h, w = image_tensor.shape[1], image_tensor.shape[2]
        return DetectionTarget(
            image=image_tensor,
            boxes=torch.zeros((0, 4), dtype=torch.float32),
            labels=torch.zeros(0, dtype=torch.long),
            masks=torch.zeros((0, h, w), dtype=torch.float32),
        )

    boxes = []
    labels = []
    masks = []

    h, w = image_tensor.shape[1], image_tensor.shape[2]

    for detection in sample.segmentations.detections:
        # Convert normalized bounding box to pixel coordinates
        bbox = detection.bounding_box  # [x, y, w, h] normalized
        x1 = int(bbox[0] * w)
        y1 = int(bbox[1] * h)
        x2 = int((bbox[0] + bbox[2]) * w)
        y2 = int((bbox[1] + bbox[3]) * h)

        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))

        # Use actual segmentation mask if available
        if hasattr(detection, "mask") and detection.mask is not None:
            boxes.append([x1, y1, x2, y2])
            # For copy-paste, we just need a unique label per detection
            # Use hash of label string to get consistent integer
            label = detection.label if hasattr(detection, "label") else "unknown"
            label_id = hash(label) % 1000  # Use hash mod 1000 for label ID
            labels.append(label_id)

            # Convert numpy boolean mask to torch tensor
            mask_np = detection.mask.astype(np.float32)

            # FiftyOne COCO masks are stored as crop regions
            # Need to place them in full image
            mask = torch.zeros((h, w), dtype=torch.float32)

            # Place the mask at the correct position using bounding box coordinates
            mask_h, mask_w = mask_np.shape
            end_y = min(y1 + mask_h, h)
            end_x = min(x1 + mask_w, w)
            actual_h = end_y - y1
            actual_w = end_x - x1

            if actual_h > 0 and actual_w > 0:
                mask_tensor = torch.from_numpy(mask_np[:actual_h, :actual_w])
                mask[y1:end_y, x1:end_x] = mask_tensor

            masks.append(mask)
        # Skip detections without masks

    # Create final masks tensor
    final_masks = (
        torch.stack(masks) if masks else torch.zeros((0, h, w), dtype=torch.float32)
    )

    return DetectionTarget(
        image=image_tensor,
        boxes=torch.tensor(boxes, dtype=torch.float32),
        labels=torch.tensor(labels, dtype=torch.long),
        masks=final_masks,
    )


def detection_target_to_fiftyone(
    target: DetectionTarget, original_filepath: str
) -> fo.Sample:
    """Convert DetectionTarget back to FiftyOne sample."""
    # Convert image tensor back to PIL format
    image_np = (target.image.permute(1, 2, 0) * 255.0).numpy().astype(np.uint8)
    image_pil = Image.fromarray(image_np)

    # Save to temporary file (in practice, you'd want to manage this better)
    import os
    import tempfile

    temp_dir = tempfile.mkdtemp()
    filename = "augmented_" + os.path.basename(original_filepath)
    temp_filepath = os.path.join(temp_dir, filename)
    image_pil.save(temp_filepath)

    # Create FiftyOne detections with masks
    detections = []
    segmentations = []
    h, w = target.image.shape[1], target.image.shape[2]

    for i in range(len(target.boxes)):
        box = target.boxes[i]
        label = target.labels[i].item()
        mask = target.masks[i]

        # Convert pixel coordinates back to normalized
        x1, y1, x2, y2 = box.tolist()
        bbox_norm = [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h]

        # Create detection without mask for detections field
        detection = fo.Detection(
            label=str(label),
            bounding_box=bbox_norm,
        )
        detections.append(detection)

        # Create detection with mask for segmentations field
        # Convert mask back to numpy
        mask_np = mask.numpy().astype(bool)

        # Extract the mask region corresponding to the bounding box
        x1_int, y1_int, x2_int, y2_int = int(x1), int(y1), int(x2), int(y2)

        # Ensure coordinates are within bounds
        x1_int = max(0, x1_int)
        y1_int = max(0, y1_int)
        x2_int = min(w, x2_int)
        y2_int = min(h, y2_int)

        if x2_int > x1_int and y2_int > y1_int:
            # Extract the mask region that corresponds to the bounding box
            mask_region = mask_np[y1_int:y2_int, x1_int:x2_int]

            # Only add if there's actually a mask (non-empty)
            if mask_region.any():
                segmentation = fo.Detection(
                    label=str(label),
                    bounding_box=bbox_norm,
                    mask=mask_region,
                )
                segmentations.append(segmentation)

    sample = fo.Sample(filepath=temp_filepath, tags=["copy-paste-augmented"])
    sample["ground_truth"] = fo.Detections(detections=detections)
    sample["segmentations"] = fo.Detections(detections=segmentations)

    return sample


def create_copy_paste_augmentation(num_samples: int = 20) -> None:
    """Create and visualize copy-paste augmented samples using proper copy-paste.

    This function demonstrates the real copy-paste augmentation functionality
    by converting FiftyOne samples to DetectionTarget format, applying the
    CopyPasteAugmentation, and converting back for visualization.
    """
    dataset = load_existing_dataset()

    # Get validation samples
    val_samples = dataset.match_tags("validation").limit(num_samples * 2)
    samples_list = list(val_samples)

    if len(samples_list) < 2:
        logging.getLogger().error("Need at least 2 samples for copy-paste augmentation")
        return

    # Create copy-paste configuration
    config = CopyPasteConfig(
        paste_probability=1.0,  # Always apply for demonstration
        max_paste_objects=3,
        min_paste_objects=1,
        scale_range=(0.5, 2.0),
    )

    # Initialize copy-paste augmentation
    copy_paste_aug = CopyPasteAugmentation(config)

    augmented_dataset = fo.Dataset("coco-copy-paste-augmented", overwrite=True)

    logging.getLogger().info("Converting FiftyOne samples to DetectionTarget format...")

    # Convert all samples to DetectionTarget format
    detection_targets = []
    for sample in samples_list:
        try:
            target = fiftyone_to_detection_target(sample)
            detection_targets.append((target, sample.filepath))
        except Exception as e:
            msg = f"Failed to convert sample {sample.filepath}: {e}"
            logging.getLogger().warning(msg)
            continue

    if len(detection_targets) < 2:
        msg = "Need at least 2 valid samples for copy-paste augmentation"
        logging.getLogger().error(msg)
        return

    msg = f"Applying copy-paste augmentation to {len(detection_targets)} samples..."
    logging.getLogger().info(msg)

    # Apply copy-paste augmentation
    for i in range(0, len(detection_targets) - 1, 2):
        target_data, target_filepath = detection_targets[i]
        source_data, _ = detection_targets[i + 1]

        # Create source objects list (other samples that can be used for pasting)
        source_objects = [source_data]

        try:
            # Apply copy-paste augmentation
            augmented_target = copy_paste_aug.transform(target_data, source_objects)

            # Convert back to FiftyOne format
            augmented_sample = detection_target_to_fiftyone(
                augmented_target, target_filepath
            )
            augmented_dataset.add_sample(augmented_sample)

            # Log the augmentation results
            orig_count = target_data.boxes.shape[0]
            aug_count = augmented_target.boxes.shape[0]
            logging.getLogger().info(
                f"Sample {i // 2}: {orig_count} -> {aug_count} objects"
            )

        except Exception as e:
            msg = f"Failed to apply copy-paste to sample {target_filepath}: {e}"
            logging.getLogger().warning(msg)
            continue

    logging.getLogger().info(
        f"Created {len(augmented_dataset)} copy-paste augmented samples"
    )

    if len(augmented_dataset) == 0:
        logging.getLogger().error("No augmented samples were created")
        return

    # Launch FiftyOne App to visualize the augmented samples
    session = fo.launch_app(augmented_dataset)
    session.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description="COCO Dataset Utilities")
    parser.add_argument(
        "mode",
        choices=["download", "visualize", "augment"],
        help="Mode to run: download dataset, visualize samples, or "
        "show copy-paste augmentation",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of samples to visualize or augment (default: 50)",
    )

    args = parser.parse_args()

    from faster_coco_eval import init_as_pycocotools

    init_as_pycocotools()

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
