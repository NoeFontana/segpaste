"""PyTorch transforms for copy-paste augmentation."""

from typing import Any, Dict, List, Union

import torch

from segpaste.augmentation import CopyPasteAugmentation
from segpaste.config import CopyPasteConfig
from segpaste.types import DetectionTarget


class CopyPasteTransform(torch.nn.Module):
    """PyTorch Transform wrapper for copy-paste augmentation.

    This transform can be used in torchvision transform pipelines.
    It expects input data in a dictionary format with keys:
    - 'image': torch.Tensor of shape [C, H, W]
    - 'boxes': torch.Tensor of shape [N, 4] in xyxy format
    - 'labels': torch.Tensor of shape [N]
    - 'masks': torch.Tensor of shape [N, H, W]
    """

    def __init__(
        self,
        source_objects: List[DetectionTarget],
        augmentation: CopyPasteAugmentation,
    ) -> None:
        """Initialize copy-paste transform.

        Args:
            source_objects: List of objects that can be pasted
            augmentation: Copy-paste augmentation instance
        """
        super().__init__()
        self.copy_paste: CopyPasteAugmentation = augmentation
        self.source_objects: List[DetectionTarget] = source_objects

    def forward(self, sample: Dict[str, Any]) -> Dict[str, DetectionTarget.TYPES]:
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
            masks=sample["masks"],
            padding_mask=sample.get("padding_mask"),
        )

        # Apply copy-paste
        augmented = self.copy_paste.transform(target_data, self.source_objects)

        # Convert back to dictionary
        result = augmented.to_dict()

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

    def __init__(self, config: CopyPasteConfig) -> None:
        """Initialize copy-paste collator.

        Args:
            config: Configuration for copy-paste augmentation
        """
        self.copy_paste: CopyPasteAugmentation = CopyPasteAugmentation(config)

    def __call__(
        self, batch: List[Dict[str, Any]]
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """Collate batch with copy-paste augmentation.

        Args:
            batch: List of sample dictionaries

        Returns:
            Collated batch dictionary
        """
        if not batch:
            return {}

        # Extract all objects from batch as potential source objects
        all_objects: List[DetectionTarget] = []
        for sample in batch:
            if "masks" in sample and sample["masks"] is not None:
                num_objects: int = sample["masks"].shape[0]
                for i in range(num_objects):
                    obj: DetectionTarget = DetectionTarget(
                        image=sample["image"],
                        boxes=sample["boxes"][i : i + 1],
                        labels=sample["labels"][i : i + 1],
                        masks=sample["masks"][i : i + 1],
                        padding_mask=sample.get("padding_mask"),
                    )
                    all_objects.append(obj)

        # Apply copy-paste to each sample
        augmented_samples: List[Dict[str, Any]] = []
        for sample in batch:
            target_data: DetectionTarget = DetectionTarget(
                image=sample["image"],
                boxes=sample["boxes"],
                labels=sample["labels"],
                masks=sample["masks"],
                padding_mask=sample.get("padding_mask"),
            )

            # Use other objects in batch as source objects
            source_objects: List[DetectionTarget] = [
                obj
                for obj in all_objects
                if not torch.equal(obj.image, target_data.image)
            ]

            if source_objects:
                augmented: DetectionTarget = self.copy_paste.transform(
                    target_data, source_objects
                )
                augmented_sample: Dict[str, DetectionTarget.TYPES] = augmented.to_dict()
                # Copy additional keys
                for key, value in sample.items():
                    if key not in augmented_sample:
                        augmented_sample[key] = value

                augmented_samples.append(augmented_sample)
            else:
                augmented_samples.append(sample)

        # Standard collation
        return self._collate_samples(augmented_samples)

    def _collate_samples(
        self, samples: List[Dict[str, Any]]
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """Collate list of samples into batch tensors.

        Args:
            samples: List of sample dictionaries

        Returns:
            Collated batch dictionary
        """
        if not samples:
            return {}

        batch: Dict[str, Union[torch.Tensor, List[torch.Tensor]]] = {}

        batch["images"] = torch.stack([sample["image"] for sample in samples])

        batch["boxes"] = [sample["boxes"] for sample in samples]
        batch["labels"] = [sample["labels"] for sample in samples]
        batch["masks"] = [sample["masks"] for sample in samples]

        return batch
