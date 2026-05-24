"""HuggingFace Mask2Former interop — ADR-0006 §6, ADR-0015 §1.

Pure-torch encoder/decoder for the ``{mask_labels, class_labels}`` dict
shape consumed by :class:`transformers.Mask2FormerImageProcessor`. No
``transformers`` import: compatibility is structural.

The COCO-panoptic ``class * MAX + instance_id`` encoding is applied only
at this boundary; the internal :class:`~segpaste.types.DenseSample`
representation keeps ``panoptic_map`` as instance ids (ADR-0006 §1).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TypedDict

import torch
from torch import Tensor
from torchvision import tv_tensors
from torchvision.ops import masks_to_boxes

from segpaste.augmentation.batch_copy_paste import BatchCopyPaste
from segpaste.integrations.torchvision import make_segpaste_collate_fn
from segpaste.presets import get_preset
from segpaste.types import (
    DenseSample,
    InstanceMask,
    PaddedBatchedDenseSample,
    PanopticMap,
    PanopticSchema,
    SemanticMap,
)


def to_hf_format(sample: DenseSample, schema: PanopticSchema) -> dict[str, Tensor]:
    """Emit ``{mask_labels, class_labels, pixel_values}`` for HF processors.

    The output shape matches what ``Mask2FormerImageProcessor.encode_inputs``
    consumes: per-instance boolean masks + an aligned class-id vector.
    ``pixel_values`` is the source image as ``float32 [C, H, W]``.
    """
    if sample.instance_masks is None or sample.panoptic_map is None:
        raise ValueError("to_hf_format requires INSTANCE and PANOPTIC modalities")
    _ = schema  # schema unused for the {mask_labels, class_labels} shape;
    # reserved for callers that want class * MAX + id encoding.
    masks = sample.instance_masks.as_subclass(Tensor).to(torch.bool)
    labels = sample.labels.to(torch.int64)
    image = sample.image.as_subclass(Tensor).to(torch.float32)
    return {
        "mask_labels": masks,
        "class_labels": labels,
        "pixel_values": image,
    }


def from_hf_format(hf: Mapping[str, Tensor], schema: PanopticSchema) -> DenseSample:
    """Reconstruct a :class:`DenseSample` from the HF dict shape.

    Assigns fresh instance ids ``1..N`` and re-derives ``panoptic_map`` and
    ``semantic_map`` from the per-instance masks + class labels + the
    thing/stuff classification in ``schema``.
    """
    masks = hf["mask_labels"].to(torch.bool)
    labels = hf["class_labels"].to(torch.int64)
    image = hf["pixel_values"].to(torch.float32)

    if masks.ndim != 3 or labels.ndim != 1 or masks.shape[0] != labels.shape[0]:
        raise ValueError(
            "mask_labels must be [N,H,W] and class_labels [N] with matching N"
        )
    n, h, w = int(masks.shape[0]), int(masks.shape[1]), int(masks.shape[2])

    semantic = torch.full((h, w), schema.ignore_index, dtype=torch.int64)
    panoptic = torch.zeros((h, w), dtype=torch.int64)
    # Assign class + fresh id in input order; later masks overwrite earlier.
    for i in range(n):
        cls = int(labels[i].item())
        mask_i = masks[i]
        if not bool(mask_i.any()):
            continue
        semantic[mask_i] = cls
        if schema.classes.get(cls) == "thing":
            panoptic[mask_i] = i + 1
        else:
            panoptic[mask_i] = 0

    if n > 0:
        boxes_t = masks_to_boxes(masks).to(torch.float32)
    else:
        boxes_t = torch.zeros((0, 4), dtype=torch.float32)

    return DenseSample(
        image=tv_tensors.Image(image),
        boxes=tv_tensors.BoundingBoxes(  # pyright: ignore[reportCallIssue]
            boxes_t, format=tv_tensors.BoundingBoxFormat.XYXY, canvas_size=(h, w)
        ),
        labels=labels,
        instance_ids=torch.arange(n, dtype=torch.int32),
        instance_masks=InstanceMask(masks),
        semantic_map=SemanticMap(semantic),
        panoptic_map=PanopticMap(panoptic),
    )


class HFBatch(TypedDict):
    """Shape emitted by :func:`to_hf_batch` and :func:`make_hf_collate_fn`."""

    mask_labels: list[Tensor]
    class_labels: list[Tensor]
    pixel_values: Tensor


HFCollateFn = Callable[[list[DenseSample]], HFBatch]


def to_hf_batch(padded: PaddedBatchedDenseSample) -> HFBatch:
    """Emit per-sample ``mask_labels`` / ``class_labels`` plus stacked images.

    Output shape:

    * ``mask_labels``: ``list[Tensor]`` of ``[n_i, H, W]`` bool, one per
      sample. Per-sample length follows ``padded.instance_valid``.
    * ``class_labels``: ``list[Tensor]`` of ``[n_i]`` int64, aligned with
      ``mask_labels``.
    * ``pixel_values``: ``Tensor[B, C, H, W]`` float32.

    Matches the training-time forward signature of
    ``Mask2FormerForUniversalSegmentation`` and the output of
    ``Mask2FormerImageProcessor.encode_inputs``.
    """
    if padded.instance_masks is None:
        raise ValueError("to_hf_batch requires instance_masks on the padded batch")
    images = padded.images.as_subclass(Tensor).to(torch.float32)
    masks_per_sample: list[Tensor] = []
    labels_per_sample: list[Tensor] = []
    for i in range(padded.images.size(0)):
        valid_i = padded.instance_valid[i]
        masks_per_sample.append(padded.instance_masks[i][valid_i].to(torch.bool))
        labels_per_sample.append(padded.labels[i][valid_i].to(torch.int64))
    return HFBatch(
        mask_labels=masks_per_sample,
        class_labels=labels_per_sample,
        pixel_values=images,
    )


def make_hf_collate_fn(preset_name: str, max_instances: int = 32) -> HFCollateFn:
    """Return a ``collate_fn`` closing the loop into the Mask2Former dict shape.

    Pipeline per batch:

    1. :meth:`BatchedDenseSample.from_samples` (ragged collate)
    2. :meth:`BatchedDenseSample.to_padded` (K-padded tensors)
    3. :class:`BatchCopyPaste` augmentation (preset-bound)
    4. :func:`to_hf_batch` (HF dict shape)

    The :class:`BatchCopyPaste` instance is constructed eagerly at factory
    call time and captured by the returned closure. With
    ``num_workers > 0`` the kernel runs on CPU inside worker processes;
    callers wanting GPU augmentation should perform steps 1-2 manually,
    move the padded batch to device, and then call
    :class:`BatchCopyPaste` + :func:`to_hf_batch`.
    """
    preset = get_preset(preset_name)
    augment = BatchCopyPaste(preset.batch_copy_paste)
    collate_padded = make_segpaste_collate_fn(max_instances)

    def _collate(samples: list[DenseSample]) -> HFBatch:
        return to_hf_batch(augment(collate_padded(samples)))

    return _collate
