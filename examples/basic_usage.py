"""Example usage of copy-paste augmentation."""

import torch
from torchvision import transforms

from segpaste import CopyPasteAugmentation, CopyPasteConfig, DetectionTarget


def create_dummy_data():
    """Create dummy detection data for demonstration."""
    # Create target image and annotations
    target_image = torch.rand(3, 512, 512)
    target_boxes = torch.tensor([[50, 50, 150, 150], [300, 300, 400, 400]])
    target_labels = torch.tensor([1, 2])

    # Create target masks
    target_masks = torch.zeros(2, 512, 512)
    for i, box in enumerate(target_boxes):
        x1, y1, x2, y2 = box.int()
        target_masks[i, y1:y2, x1:x2] = 1.0

    target_data = DetectionTarget(
        image=target_image, boxes=target_boxes, labels=target_labels, masks=target_masks
    )

    # Create source objects to paste
    source_objects = []
    for i in range(5):
        source_image = torch.rand(3, 100, 100)
        source_box = torch.tensor([[10, 10, 80, 80]])
        source_label = torch.tensor([i + 3])  # Different classes

        source_mask = torch.zeros(1, 100, 100)
        source_mask[0, 10:80, 10:80] = 1.0

        source_obj = DetectionTarget(
            image=source_image, boxes=source_box, labels=source_label, masks=source_mask
        )
        source_objects.append(source_obj)

    return target_data, source_objects


def main():
    """Demonstrate copy-paste augmentation usage."""
    print("Copy-Paste Augmentation Example")
    print("=" * 40)

    # Create dummy data
    target_data, source_objects = create_dummy_data()

    print(f"Original target image shape: {target_data.image.shape}")
    print(f"Original target boxes: {target_data.boxes.shape[0]} objects")
    print(f"Source objects available: {len(source_objects)}")

    # Create copy-paste configuration
    config = CopyPasteConfig(
        paste_probability=1.0,  # Always apply for demo
        max_paste_objects=3,
        min_paste_objects=1,
        scale_range=(0.5, 1.5),
        blend_mode="gaussian",
        occluded_area_threshold=0.3,
        enable_flip=True,
    )

    # Initialize copy-paste augmentation
    copy_paste = CopyPasteAugmentation(config)

    # Apply augmentation
    print("\nApplying copy-paste augmentation...")
    augmented_data = copy_paste(target_data, source_objects)

    print(f"Augmented image shape: {augmented_data.image.shape}")
    print(f"Augmented boxes: {augmented_data.boxes.shape[0]} objects")
    print(
        f"Objects added: {augmented_data.boxes.shape[0] - target_data.boxes.shape[0]}"
    )

    # Show before/after comparison
    print("\nBefore augmentation:")
    print(f"  - Boxes: {target_data.boxes}")
    print(f"  - Labels: {target_data.labels}")

    print("\nAfter augmentation:")
    print(f"  - Boxes: {augmented_data.boxes}")
    print(f"  - Labels: {augmented_data.labels}")

    # Example with torchvision transforms
    print("\nUsing with torchvision transforms:")
    from segpaste import CopyPasteTransform

    # Create transform
    copy_paste_transform = CopyPasteTransform(
        source_objects=source_objects, config=config
    )

    # Convert to dictionary format (as expected by torchvision)
    sample = {
        "image": target_data.image,
        "boxes": target_data.boxes,
        "labels": target_data.labels,
        "masks": target_data.masks,
        "image_id": torch.tensor([1]),  # Additional metadata
    }

    # Apply transform
    transformed_sample = copy_paste_transform(sample)

    print(f"Transformed sample keys: {list(transformed_sample.keys())}")
    print(f"Transformed image shape: {transformed_sample['image'].shape}")
    print(f"Transformed objects: {transformed_sample['boxes'].shape[0]}")

    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
