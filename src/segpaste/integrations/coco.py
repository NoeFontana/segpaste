import os
from typing import Any, Callable, Dict, List, Tuple

import torch
import torchvision
from faster_coco_eval import COCO
from faster_coco_eval import mask as coco_mask
from torchvision import tv_tensors
from torchvision.datasets import VisionDataset
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F

from segpaste.types.data_structures import PaddingMask


def segmentation_to_mask(
    segmentation: Any, canvas_size: tuple[int, int]
) -> torch.Tensor:
    """Convert COCO segmentation to binary mask.

    Args:
        segmentation (ValidRleType): COCO segmentation (RLE or polygon format).
        canvas_size (tuple[int, int]): Size of the canvas (height, width).

    Returns:
        torch.Tensor: Binary mask tensor.
    """
    # TODO: Clean that up
    if not isinstance(segmentation, (dict, list)):
        raise ValueError(
            f"COCO segmentation expected to be dict or list, got {type(segmentation)}"
        )

    h, w = canvas_size
    if isinstance(segmentation, dict):
        # if counts is a string, it is already an encoded RLE mask
        if not isinstance(segmentation["counts"], str):
            segmentation = coco_mask.frPyObjects(segmentation, h, w)
    elif isinstance(segmentation, list):
        segmentation = coco_mask.merge(coco_mask.frPyObjects(segmentation, h, w))  # pyright: ignore[reportArgumentType]
    return torch.from_numpy(coco_mask.decode(segmentation))


class CocoDetectionV2(VisionDataset):  # type: ignore[misc]
    def __init__(
        self,
        image_folder: str,
        label_path: str,
        transforms: Callable | None = None,  # type: ignore[type-arg]
        target_keys: list[str] | None = None,
    ):
        super().__init__(root=image_folder, transforms=transforms)

        self.label_path = label_path
        self.coco = COCO(label_path)

        # Filter the list, keeping only IDs that have annotations
        self.valid_img_ids = sorted(
            img_id
            for img_id in self.coco.get_img_ids()
            if len(self.coco.get_ann_ids([img_id])) > 0
        )

        if target_keys is None:
            target_keys = ["image_id", "padding_mask", "boxes", "labels", "masks"]
        self.target_keys = target_keys

    def __len__(self) -> int:
        return len(self.valid_img_ids)

    def _load_image(self, id: int) -> torch.Tensor:
        path = os.path.join(self.root, self.coco.loadImgs(id)[0]["file_name"])
        image: torch.Tensor = torchvision.io.decode_image(
            path, mode=torchvision.io.ImageReadMode.RGB
        )
        return image

    def _load_target(self, id: int) -> list[dict[str, Any]]:
        target: list[dict[str, Any]] = self.coco.loadAnns(self.coco.getAnnIds([id]))
        return target

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        if not isinstance(index, int):
            raise ValueError(
                f"Index must be of type integer, got {type(index)} instead."
            )

        image_id = self.valid_img_ids[index]
        image = self._load_image(image_id)
        target_list = self._load_target(image_id)

        canvas_size = image.shape[-2:]  # (H, W)

        # Convert list of annotation dicts to the format expected by transforms v2
        if not target_list:
            # Handle empty targets
            target = {
                "image_id": image_id,
                "boxes": tv_tensors.BoundingBoxes(
                    torch.zeros((0, 4)),
                    format=tv_tensors.BoundingBoxFormat.XYXY,
                    canvas_size=canvas_size,
                    dtype=torch.float32,
                ),  # pyright: ignore[reportCallIssue]
                "labels": torch.zeros(0, dtype=torch.long),
                "masks": tv_tensors.Mask(
                    torch.zeros((0, *canvas_size), dtype=torch.uint8)
                ),
            }
        else:
            # Extract bounding boxes and convert from XYWH to XYXY
            boxes = torch.tensor(
                [ann["bbox"] for ann in target_list], dtype=torch.float32
            )
            boxes = F.convert_bounding_box_format(
                tv_tensors.BoundingBoxes(
                    data=boxes,
                    format=tv_tensors.BoundingBoxFormat.XYWH,
                    canvas_size=canvas_size,
                ),  # pyright: ignore[reportCallIssue]
                new_format=tv_tensors.BoundingBoxFormat.XYXY,
            )

            # Extract labels
            labels = torch.tensor(
                [ann["category_id"] for ann in target_list], dtype=torch.long
            )
            target = {
                "image_id": image_id,
                "boxes": boxes,
                "labels": labels,
            }
            # Convert segmentations to masks if available
            if "masks" in self.target_keys:
                masks = []
                for ann in target_list:
                    segmentation = ann["segmentation"]
                    mask = segmentation_to_mask(
                        segmentation=segmentation, canvas_size=canvas_size
                    )
                    masks.append(mask)

                masks_tensor = tv_tensors.Mask(torch.stack(masks))
                target["masks"] = masks_tensor

        if "padding_mask" in self.target_keys:
            # 1xHxW mask of ones, since all pixels are valid at this point
            target["padding_mask"] = PaddingMask(
                torch.zeros((1, *canvas_size), dtype=torch.bool)
            )

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


def labels_getter(
    sample: tuple[tv_tensors.Image, Dict[str, tv_tensors.TVTensor]],
) -> tuple[tv_tensors.BoundingBoxes, tv_tensors.Mask, torch.Tensor]:
    """Extract labels tensor from the sample structure."""
    target = sample[1]

    # TODO: Handle the case where there are no instances
    return (target["boxes"], target["masks"], target["labels"])  # pyright: ignore[reportReturnType]


def add_image_collate_fn(
    batch: List[Tuple[tv_tensors.Image, Dict[str, Any]]],
) -> List[Dict[str, torch.Tensor]]:
    """Custom collate function to convert COCO annotations to expected format.

    Converts a batch of (image, target) into a list of dictionaries
        containing the 'image' and all target keys.
    """
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


def create_coco_dataloader(
    image_folder: str,
    label_path: str,
    transforms: v2.Transform,
    batch_size: int = 4,
    collate_fn: Any = add_image_collate_fn,
) -> torch.utils.data.DataLoader[tuple[Any, Any]]:
    """Create COCO dataset and dataloader for testing.

    Args:
        dataset_path: Path to COCO dataset directory
        batch_size: Batch size for dataloader
        collate_fn: Collate function for dataloader

    Returns:
        DataLoader for COCO dataset
    """

    dataset = CocoDetectionV2(
        image_folder=image_folder,
        label_path=label_path,
        transforms=transforms,
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
