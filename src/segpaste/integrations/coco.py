import json
import os
from collections.abc import Callable
from typing import Any, cast

import torch
import torchvision
from faster_coco_eval import COCO
from faster_coco_eval import mask as coco_mask
from torchvision import tv_tensors
from torchvision.datasets import VisionDataset
from torchvision.ops import masks_to_boxes
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F

from segpaste.types import DenseSample, InstanceMask, PanopticMap, SemanticMap
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
                    torch.zeros((0, *canvas_size), dtype=torch.bool)
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

                masks_tensor = tv_tensors.Mask(torch.stack(masks).to(torch.bool))
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


class CocoPanopticV2(VisionDataset):
    """COCO panoptic loader.

    Reads ``panoptic_*.json`` (per-image ``segments_info`` records) plus the
    PNG-encoded panoptic maps under ``panoptic_*/`` and emits a
    :class:`DenseSample` carrying ``image``, ``instance_masks`` (thing
    instances only), ``labels``, ``instance_ids``, ``boxes``, ``semantic_map``,
    ``panoptic_map`` and a zero-valued ``padding_mask``.

    Per ADR-0001 §(ii), ``panoptic_map`` encodes stuff pixels as ``0`` and
    every thing instance with a unique non-zero id; ``semantic_map`` carries
    the COCO category id per pixel (with ``schema.ignore_index`` on
    unlabelled void).
    """

    def __init__(
        self,
        image_folder: str,
        panoptic_folder: str,
        label_path: str,
        transforms: Callable | None = None,  # pyright: ignore[reportMissingTypeArgument, reportUnknownParameterType]
        ignore_index: int = 255,
    ):
        super().__init__(root=image_folder, transforms=transforms)
        with open(label_path) as f:
            data: dict[str, Any] = json.load(f)
        self._panoptic_folder = panoptic_folder
        self._ignore_index = ignore_index
        self._categories: dict[int, dict[str, Any]] = {
            int(c["id"]): c for c in data["categories"]
        }
        self._images: dict[int, dict[str, Any]] = {
            int(img["id"]): img for img in data["images"]
        }
        self._annotations: dict[int, dict[str, Any]] = {
            int(a["image_id"]): a for a in data["annotations"]
        }
        self._image_ids: list[int] = sorted(self._annotations.keys())

    def __len__(self) -> int:
        return len(self._image_ids)

    def __getitem__(self, index: int) -> DenseSample:
        if not isinstance(index, int):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ValueError(
                f"Index must be of type integer, got {type(index)} instead."
            )

        image_id = self._image_ids[index]
        img_meta = self._images[image_id]
        ann = self._annotations[image_id]

        image: torch.Tensor = torchvision.io.decode_image(
            os.path.join(self.root, img_meta["file_name"]),
            mode=torchvision.io.ImageReadMode.RGB,
        )
        pan_rgb = torchvision.io.decode_image(
            os.path.join(self._panoptic_folder, ann["file_name"]),
            mode=torchvision.io.ImageReadMode.RGB,
        ).to(torch.int64)
        # Per the COCO panoptic spec (https://cocodataset.org/#panoptic-2018),
        # segment ids are encoded into the PNG channels as id = R + 256*G + 256^2*B.
        seg_id = pan_rgb[0] + pan_rgb[1] * 256 + pan_rgb[2] * (256 * 256)

        h, w = seg_id.shape
        canvas_size = (h, w)
        semantic = torch.full((h, w), self._ignore_index, dtype=torch.int64)
        panoptic = torch.zeros((h, w), dtype=torch.int64)
        thing_masks: list[torch.Tensor] = []
        thing_labels: list[int] = []
        next_thing_id = 1
        for seg in ann["segments_info"]:
            sid = int(seg["id"])
            cat_id = int(seg["category_id"])
            mask = seg_id == sid
            semantic[mask] = cat_id
            if int(self._categories[cat_id].get("isthing", 0)) == 1:
                panoptic[mask] = next_thing_id
                thing_masks.append(mask)
                thing_labels.append(cat_id)
                next_thing_id += 1

        if thing_masks:
            masks_t = torch.stack(thing_masks).to(torch.bool)
            labels_t = torch.tensor(thing_labels, dtype=torch.int64)
            boxes_t = masks_to_boxes(masks_t).to(torch.float32)
        else:
            masks_t = torch.zeros((0, h, w), dtype=torch.bool)
            labels_t = torch.zeros((0,), dtype=torch.int64)
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)

        target: dict[str, Any] = {
            "image_id": image_id,
            "boxes": tv_tensors.BoundingBoxes(  # pyright: ignore[reportCallIssue]
                boxes_t,
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=canvas_size,
            ),
            "labels": labels_t,
            "masks": tv_tensors.Mask(masks_t),
            "semantic_map": SemanticMap(semantic),
            "panoptic_map": PanopticMap(panoptic),
            "padding_mask": PaddingMask(
                torch.zeros((1, *canvas_size), dtype=torch.bool)
            ),
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        labels_tensor = cast(torch.Tensor, target["labels"])
        boxes_tv = cast(tv_tensors.BoundingBoxes, target["boxes"])
        masks_tv = cast(tv_tensors.Mask, target["masks"])
        semantic_tv = cast(torch.Tensor, target["semantic_map"])
        panoptic_tv = cast(torch.Tensor, target["panoptic_map"])
        padding_tv = cast(PaddingMask, target["padding_mask"])

        wrapped_image = (
            image if isinstance(image, tv_tensors.Image) else tv_tensors.Image(image)
        )
        instance_count = labels_tensor.size(0)
        return DenseSample(
            image=wrapped_image,
            boxes=boxes_tv,
            labels=labels_tensor,
            instance_ids=torch.arange(instance_count, dtype=torch.int32),
            instance_masks=InstanceMask(masks_tv.to(torch.bool)),
            semantic_map=SemanticMap(semantic_tv.to(torch.int64)),
            panoptic_map=PanopticMap(panoptic_tv.to(torch.int64)),
            padding_mask=padding_tv,
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

    Downstream code calls :meth:`BatchedDenseSample.from_samples` followed by
    :meth:`BatchedDenseSample.to_padded` to feed :class:`BatchCopyPaste`.
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
            yields ``list[DenseSample]`` — wrap the result through
            :meth:`BatchedDenseSample.from_samples` → ``.to_padded(K)`` →
            :class:`BatchCopyPaste` to apply augmentation.

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
