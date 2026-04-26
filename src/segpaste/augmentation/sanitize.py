"""Mask-based instance sanitization for transforms-v2 pipelines.

The torchvision-stock :class:`SanitizeBoundingBoxes` only inspects bbox
extents (``min_size`` / ``min_area``) and drops instances whose box has
collapsed below threshold. This is too coarse for instance segmentation:
a heavily occluded object whose mask is still visible can have its box
clamped to the canvas edge and disappear, while a sliver-mask instance
whose box happens to look fine survives.

:class:`SanitizeInstances` filters by post-augmentation mask area and
recomputes bboxes from the surviving masks via
:func:`torchvision.ops.masks_to_boxes`. ``PaddingMask`` is forwarded
unchanged because it is per-image, not per-instance.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import Tensor
from torch.utils._pytree import tree_flatten, tree_unflatten
from torchvision import tv_tensors
from torchvision.ops import masks_to_boxes
from torchvision.transforms.v2 import Transform
from torchvision.transforms.v2 import functional as F

from segpaste.types.data_structures import PaddingMask


class SanitizeInstances(Transform):
    """Filter instances by mask area; recompute bboxes from cleaned masks.

    Drops instances whose binary mask has fewer than ``min_mask_area``
    True pixels post-augmentation, then replaces each surviving instance's
    bbox with the tight box of its mask via
    :func:`torchvision.ops.masks_to_boxes`. When no per-instance mask is
    present in the input, falls back to bbox-extent filtering.

    Parameters
    ----------
    min_mask_area
        Minimum number of True pixels a mask must contain to survive.
        ``10`` matches the lower-bound used by Detectron2's
        ``filter_empty_instances`` ergonomics for COCO-scale crops.
    labels_getter
        Callable receiving the input structure (the same tuple/dict
        passed into the transform's ``__call__``) and returning the
        instance-aligned tensors that should be co-filtered with the
        boxes — canonically ``(boxes, masks, labels)``. Pass ``None`` to
        disable label filtering when the structure has no labels.
    """

    def __init__(
        self,
        *,
        min_mask_area: int = 10,
        labels_getter: Callable[[Any], Sequence[Tensor] | None] | None = None,
    ) -> None:
        super().__init__()
        if min_mask_area < 0:
            raise ValueError(f"min_mask_area must be non-negative, got {min_mask_area}")
        self.min_mask_area = int(min_mask_area)
        self._labels_getter = labels_getter

    def forward(self, *inputs: Any) -> Any:
        if len(inputs) == 0:
            return inputs
        sample: Any = inputs if len(inputs) > 1 else inputs[0]
        labels = (
            self._labels_getter(sample) if self._labels_getter is not None else None
        )
        flat_inputs, spec = tree_flatten(sample)

        masks: tv_tensors.Mask | None = next(
            (
                x
                for x in flat_inputs
                if isinstance(x, tv_tensors.Mask) and not isinstance(x, PaddingMask)
            ),
            None,
        )
        boxes: tv_tensors.BoundingBoxes | None = next(
            (x for x in flat_inputs if isinstance(x, tv_tensors.BoundingBoxes)),
            None,
        )

        bool_masks: Tensor | None = None
        if masks is not None and masks.shape[0] > 0:
            bool_masks = masks.as_subclass(torch.Tensor).to(torch.bool)
            areas = bool_masks.flatten(1).sum(dim=1)
            valid = areas >= self.min_mask_area
        elif boxes is not None and boxes.shape[0] > 0:
            box_t = boxes.as_subclass(torch.Tensor)
            valid = ((box_t[:, 2] - box_t[:, 0]) > 0) & (
                (box_t[:, 3] - box_t[:, 1]) > 0
            )
        else:
            valid = torch.zeros(0, dtype=torch.bool)

        recomputed_xyxy: Tensor | None = None
        if bool_masks is not None and bool(valid.any()):
            recomputed_xyxy = masks_to_boxes(bool_masks[valid])

        flat_outputs = [
            self._filter_one(x, valid, boxes, labels, recomputed_xyxy)
            for x in flat_inputs
        ]
        return tree_unflatten(flat_outputs, spec)

    @staticmethod
    def _filter_one(
        inpt: Any,
        valid: Tensor,
        boxes: tv_tensors.BoundingBoxes | None,
        labels: Sequence[Tensor] | None,
        recomputed_xyxy: Tensor | None,
    ) -> Any:
        if isinstance(inpt, PaddingMask):
            return inpt
        if inpt is boxes:
            if recomputed_xyxy is None:
                return tv_tensors.wrap(inpt[valid], like=inpt)
            new_boxes = tv_tensors.BoundingBoxes(  # pyright: ignore[reportCallIssue]
                recomputed_xyxy.to(dtype=inpt.dtype),
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=inpt.canvas_size,
            )
            return F.convert_bounding_box_format(new_boxes, new_format=inpt.format)
        if isinstance(inpt, tv_tensors.Mask):
            if inpt.ndim < 3:
                return inpt
            return tv_tensors.wrap(inpt[valid], like=inpt)
        if labels is not None and any(inpt is x for x in labels):
            return inpt[valid]
        return inpt
