import logging
from typing import Any, Callable, Dict, List, Tuple

import torch
from torchvision import tv_tensors
from torchvision.datasets import CocoDetection
from torchvision.transforms import v2
from torchvision.tv_tensors._dataset_wrapper import wrap_dataset_for_transforms_v2


class FilteredCocoDetection(CocoDetection):  # type: ignore[misc]
    def __init__(
        self,
        image_folder: str,
        label_path: str,
        transforms: Callable | None = None,  # type: ignore[type-arg]
    ):
        # First, initialize the parent class
        super().__init__(root=image_folder, annFile=label_path, transforms=transforms)

        # Get the original list of all image IDs
        all_ids = self.coco.getImgIds()

        # Filter the list, keeping only IDs that have annotations
        self.ids = [
            img_id for img_id in all_ids if len(self.coco.getAnnIds(imgIds=img_id)) > 0
        ]
        logger = logging.getLogger(__name__)
        logger.info(
            f"Original number of images: {len(all_ids)}. "
            f"Number of images with annotations: {len(self.ids)}"
        )

    def __len__(self) -> int:
        return len(self.ids)


def labels_getter(
    sample: tuple[tv_tensors.Image, Dict[str, tv_tensors.TVTensor]],
) -> tuple[tv_tensors.BoundingBoxes, tv_tensors.Mask, torch.Tensor]:
    """Extract labels tensor from the sample structure."""
    target = sample[1]

    # TODO: Handle the case where there are no instances
    return (target["boxes"], target["masks"], target["labels"])  # pyright: ignore[reportReturnType]


def create_coco_dataset(
    image_folder: str, label_path: str, transforms: v2.Transform, batch_size: int = 4
) -> torch.utils.data.DataLoader:  # type: ignore[type-arg]
    """Create COCO dataset and dataloader for testing.

    Args:
        dataset_path: Path to COCO dataset directory
        batch_size: Batch size for dataloader

    Returns:
        DataLoader for COCO dataset
    """

    def coco_collate_fn(
        batch: List[Tuple[tv_tensors.Image, Dict[str, Any]]],
    ) -> List[Dict[str, torch.Tensor]]:
        """Custom collate function to convert COCO annotations to expected format."""
        samples: List[Dict[str, torch.Tensor]] = []

        for image, targets in batch:
            sample = targets.copy()
            sample.update(
                {
                    "image": image,
                }
            )
            samples.append(sample)
        return samples

    dataset = wrap_dataset_for_transforms_v2(
        FilteredCocoDetection(
            image_folder=image_folder,
            label_path=label_path,
            transforms=transforms,
        ),
        target_keys=["image_id", "boxes", "labels", "masks"],
    )

    return torch.utils.data.DataLoader(
        dataset,  # pyright: ignore[reportArgumentType]
        batch_size=batch_size,
        shuffle=False,
        collate_fn=coco_collate_fn,
    )
