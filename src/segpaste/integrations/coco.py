import os
from collections.abc import Callable
from typing import Any, cast

import torch
import torchvision
from faster_coco_eval import COCO
from faster_coco_eval import mask as coco_mask
from torchvision import tv_tensors
from torchvision.datasets import VisionDataset
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F

from segpaste.types import DenseSample, InstanceMask
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
    if not isinstance(segmentation, dict | list):
        raise ValueError(
            f"COCO segmentation expected to be dict or list, got {type(segmentation)}"
        )

    h, w = canvas_size
    if isinstance(segmentation, dict):
        # if counts is a string, it is already an encoded RLE mask
        if not isinstance(segmentation["counts"], str):
            segmentation = coco_mask.frPyObjects(segmentation, h, w)
    else:
        segmentation = coco_mask.merge(coco_mask.frPyObjects(segmentation, h, w))  # pyright: ignore[reportArgumentType]
    return torch.from_numpy(coco_mask.decode(segmentation))


class CocoDetectionV2(VisionDataset):
    def __init__(
        self,
        image_folder: str,
        label_path: str,
        transforms: Callable | None = None,  # pyright: ignore[reportMissingTypeArgument, reportUnknownParameterType]
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

    def __getitem__(self, index: int) -> DenseSample:
        if not isinstance(index, int):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ValueError(
                f"Index must be of type integer, got {type(index)} instead."
            )

        image_id = self.valid_img_ids[index]
        image = self._load_image(image_id)
        target_list = self._load_target(image_id)

        canvas_size = image.shape[-2], image.shape[-1]  # (H, W)

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

        labels_tensor = cast(torch.Tensor, target["labels"])
        boxes_tv = cast(tv_tensors.BoundingBoxes, target["boxes"])

        masks_val = target.get("masks")
        if masks_val is not None:
            instance_masks = InstanceMask(cast(torch.Tensor, masks_val).to(torch.bool))
            instance_ids = torch.arange(labels_tensor.size(0), dtype=torch.int32)
        else:
            instance_masks = None
            instance_ids = None

        padding_mask_val = target.get("padding_mask")
        padding_mask: PaddingMask | None = (
            cast(PaddingMask, padding_mask_val)
            if padding_mask_val is not None
            else None
        )

        wrapped_image = (
            image if isinstance(image, tv_tensors.Image) else tv_tensors.Image(image)
        )
        return DenseSample(
            image=wrapped_image,
            boxes=boxes_tv,
            labels=labels_tensor,
            instance_ids=instance_ids,
            instance_masks=instance_masks,
            padding_mask=padding_mask,
        )


def labels_getter(
    sample: tuple[tv_tensors.Image, dict[str, tv_tensors.TVTensor]],
) -> tuple[tv_tensors.BoundingBoxes, tv_tensors.Mask, torch.Tensor]:
    """Extract labels tensor from the sample structure."""
    target = sample[1]

    # TODO: Handle the case where there are no instances
    return (target["boxes"], target["masks"], target["labels"])  # pyright: ignore[reportReturnType]


def _identity_collate(batch: list[DenseSample]) -> list[DenseSample]:
    """Default collate that passes ``list[DenseSample]`` through unchanged.

    :class:`segpaste.augmentation.CopyPasteCollator` is the intended collator;
    when absent, downstream code calls :meth:`BatchedDenseSample.from_samples`.
    """
    return batch


def create_coco_dataloader(
    image_folder: str,
    label_path: str,
    transforms: v2.Transform,
    batch_size: int = 4,
    collate_fn: Any = _identity_collate,
) -> torch.utils.data.DataLoader[DenseSample]:
    """Create a COCO DataLoader preconfigured for segpaste pipelines.

    Args:
        image_folder (str): Directory containing the COCO image files.
        label_path (str): Path to the COCO JSON annotations file.
        transforms (v2.Transform): Transform applied to each sample.
        batch_size (int): Batch size for the returned DataLoader.
        collate_fn: Collate function; defaults to an identity collate that
            yields ``list[DenseSample]`` — wire :class:`CopyPasteCollator`
            when copy-paste augmentation is desired.

    Returns:
        A DataLoader yielding :class:`DenseSample` instances.
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
