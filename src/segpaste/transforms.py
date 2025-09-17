"""PyTorch transforms for copy-paste augmentation."""

from typing import Any, Dict, List

import torch
from torchvision.transforms.v2 import Transform

from .copy_paste import CopyPasteAugmentation
from .data_types import CopyPasteConfig, DetectionTarget


class CopyPasteTransform(Transform):
    """PyTorch Transform wrapper for copy-paste augmentation.

    This transform can be used in torchvision transform pipelines.
    It expects input data in a dictionary format with keys:
    - 'image': torch.Tensor of shape [C, H, W]
    - 'boxes': torch.Tensor of shape [N, 4] in xyxy format
    - 'labels': torch.Tensor of shape [N]
    - 'masks': torch.Tensor of shape [N, H, W] (optional)
    """

    def __init__(
        self,
        source_objects: List[DetectionTarget],
        config: CopyPasteConfig | None = None,
    ) -> None:
        """Initialize copy-paste transform.

        Args:
            source_objects: List of objects that can be pasted
            config: Configuration for copy-paste augmentation
        """
        super().__init__()
        self.copy_paste = CopyPasteAugmentation(config)
        self.source_objects = source_objects

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Apply copy-paste augmentation to sample.

        Args:
            sample: Dictionary containing image and annotations

        Returns:
            Augmented sample dictionary
        """
        # Convert to DetectionTarget
        target_data = DetectionTarget(
            image=sample["image"],
            boxes=sample["boxes"],
            labels=sample["labels"],
            masks=sample.get("masks"),
        )

        # Apply copy-paste
        augmented = self.copy_paste(target_data, self.source_objects)

        # Convert back to dictionary
        result = {
            "image": augmented.image,
            "boxes": augmented.boxes,
            "labels": augmented.labels,
        }

        if augmented.masks is not None:
            result["masks"] = augmented.masks

        # Copy any additional keys from original sample
        for key, value in sample.items():
            if key not in result:
                result[key] = value

        return result

    def __repr__(self) -> str:
        """Return string representation of transform."""
        return (
            f"{self.__class__.__name__}("
            f"num_source_objects={len(self.source_objects)}, "
            f"config={self.copy_paste.config})"
        )


class CopyPasteCollator:
    """Collate function for batching data with copy-paste augmentation.

    This collator applies copy-paste augmentation at batch time,
    allowing objects from different images in the batch to be used
    as source objects for pasting.
    """

    def __init__(self, config: CopyPasteConfig | None = None):
        """Initialize copy-paste collator.

        Args:
            config: Configuration for copy-paste augmentation
        """
        self.copy_paste = CopyPasteAugmentation(config)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate batch with copy-paste augmentation.

        Args:
            batch: List of sample dictionaries

        Returns:
            Collated batch dictionary
        """
        if not batch:
            return {}

        # Extract all objects from batch as potential source objects
        all_objects = []
        for sample in batch:
            if "masks" in sample and sample["masks"] is not None:
                num_objects = sample["masks"].shape[0]
                for i in range(num_objects):
                    obj = DetectionTarget(
                        image=sample["image"],
                        boxes=sample["boxes"][i : i + 1],
                        labels=sample["labels"][i : i + 1],
                        masks=sample["masks"][i : i + 1],
                    )
                    all_objects.append(obj)

        # Apply copy-paste to each sample
        augmented_samples = []
        for sample in batch:
            target_data = DetectionTarget(
                image=sample["image"],
                boxes=sample["boxes"],
                labels=sample["labels"],
                masks=sample.get("masks"),
            )

            # Use other objects in batch as source objects
            source_objects = [
                obj
                for obj in all_objects
                if not torch.equal(obj.image, target_data.image)
            ]

            if source_objects:
                augmented = self.copy_paste(target_data, source_objects)
                augmented_sample = {
                    "image": augmented.image,
                    "boxes": augmented.boxes,
                    "labels": augmented.labels,
                }
                if augmented.masks is not None:
                    augmented_sample["masks"] = augmented.masks

                # Copy additional keys
                for key, value in sample.items():
                    if key not in augmented_sample:
                        augmented_sample[key] = value

                augmented_samples.append(augmented_sample)
            else:
                augmented_samples.append(sample)

        # Standard collation
        return self._collate_samples(augmented_samples)

    def _collate_samples(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate list of samples into batch tensors.

        Args:
            samples: List of sample dictionaries

        Returns:
            Collated batch dictionary
        """
        if not samples:
            return {}

        batch = {}

        # Stack images
        batch["images"] = torch.stack([sample["image"] for sample in samples])

        # Collect variable-length tensors (boxes, labels, masks)
        batch["boxes"] = [sample["boxes"] for sample in samples]
        batch["labels"] = [sample["labels"] for sample in samples]

        if "masks" in samples[0] and samples[0]["masks"] is not None:
            batch["masks"] = [sample["masks"] for sample in samples]

        # Handle additional keys
        for key in samples[0]:
            if key not in ["image", "boxes", "labels", "masks"]:
                if isinstance(samples[0][key], torch.Tensor):
                    try:
                        batch[key] = torch.stack([sample[key] for sample in samples])
                    except RuntimeError:
                        # Variable size tensors
                        batch[key] = [sample[key] for sample in samples]
                else:
                    batch[key] = [sample[key] for sample in samples]

        return batch
