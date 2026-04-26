"""Tests for :class:`SanitizeInstances`."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torchvision import tv_tensors

from segpaste.augmentation import SanitizeInstances
from segpaste.integrations import labels_getter
from segpaste.types.data_structures import PaddingMask

H = W = 64


def _build_sample(
    *,
    mask_areas: list[int],
    boxes: list[list[float]] | None = None,
) -> tuple[tv_tensors.Image, dict[str, Any]]:
    n = len(mask_areas)
    image = tv_tensors.Image(torch.zeros(3, H, W, dtype=torch.uint8))
    masks = torch.zeros(n, H, W, dtype=torch.bool)
    raw_boxes: list[list[float]] = []
    for i, area in enumerate(mask_areas):
        side = max(1, round(area**0.5))
        x1, y1 = i * 8, i * 8
        x2 = min(W, x1 + side)
        y2 = min(H, y1 + side)
        masks[i, y1:y2, x1:x2] = True
        raw_boxes.append([float(x1), float(y1), float(x2), float(y2)])
    if boxes is not None:
        raw_boxes = boxes
    target = {
        "boxes": tv_tensors.BoundingBoxes(  # pyright: ignore[reportCallIssue]
            torch.tensor(raw_boxes, dtype=torch.float32),
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=(H, W),
        ),
        "masks": tv_tensors.Mask(masks),
        "labels": torch.arange(1, n + 1, dtype=torch.int64),
    }
    return image, target


class TestMaskAreaFilter:
    def test_drops_below_threshold(self) -> None:
        image, target = _build_sample(mask_areas=[100, 5, 200, 1])
        sanitize = SanitizeInstances(min_mask_area=10, labels_getter=labels_getter)
        _, out_target = sanitize(image, target)

        assert out_target["masks"].shape[0] == 2
        assert out_target["boxes"].shape[0] == 2
        assert out_target["labels"].tolist() == [1, 3]

    def test_keeps_all_when_all_above_threshold(self) -> None:
        image, target = _build_sample(mask_areas=[100, 200, 300])
        sanitize = SanitizeInstances(min_mask_area=10, labels_getter=labels_getter)
        _, out_target = sanitize(image, target)

        assert out_target["masks"].shape[0] == 3
        assert out_target["labels"].tolist() == [1, 2, 3]

    def test_drops_all_when_all_below_threshold(self) -> None:
        image, target = _build_sample(mask_areas=[2, 3])
        sanitize = SanitizeInstances(min_mask_area=10, labels_getter=labels_getter)
        _, out_target = sanitize(image, target)

        assert out_target["masks"].shape[0] == 0
        assert out_target["boxes"].shape[0] == 0
        assert out_target["labels"].shape[0] == 0


class TestBoxRecomputation:
    def test_recomputes_box_from_mask(self) -> None:
        """Box clamped to canvas edge but mask still visible → recover via mask."""
        image, target = _build_sample(
            mask_areas=[100],
            boxes=[[63.0, 30.0, 64.0, 31.0]],
        )
        sanitize = SanitizeInstances(min_mask_area=10, labels_getter=labels_getter)
        _, out_target = sanitize(image, target)

        assert out_target["boxes"].shape[0] == 1
        recomputed = out_target["boxes"][0].tolist()
        # masks_to_boxes returns [x_min, y_min, x_max, y_max] with inclusive
        # max indices; a 10x10 True region at top-left gives x_max=y_max=9.
        assert recomputed == [0.0, 0.0, 9.0, 9.0]

    def test_box_format_preserved(self) -> None:
        image, target = _build_sample(mask_areas=[100, 200])
        target["boxes"] = tv_tensors.wrap(  # pyright: ignore[reportCallIssue]
            target["boxes"],
            like=target["boxes"],
            format=tv_tensors.BoundingBoxFormat.CXCYWH,
        )
        from torchvision.transforms.v2 import functional as F

        target["boxes"] = F.convert_bounding_box_format(
            target["boxes"], new_format=tv_tensors.BoundingBoxFormat.CXCYWH
        )
        sanitize = SanitizeInstances(min_mask_area=10, labels_getter=labels_getter)
        _, out_target = sanitize(image, target)

        assert out_target["boxes"].format == tv_tensors.BoundingBoxFormat.CXCYWH


class TestPaddingMaskPassthrough:
    def test_padding_mask_unchanged(self) -> None:
        image, target = _build_sample(mask_areas=[100, 5, 200])
        target["padding_mask"] = PaddingMask(torch.zeros(1, H, W, dtype=torch.bool))
        sanitize = SanitizeInstances(min_mask_area=10, labels_getter=labels_getter)
        _, out_target = sanitize(image, target)

        assert out_target["padding_mask"].shape == (1, H, W)
        assert out_target["padding_mask"].dtype == torch.bool


class TestEdgeCases:
    def test_empty_input(self) -> None:
        image = tv_tensors.Image(torch.zeros(3, H, W, dtype=torch.uint8))
        target = {
            "boxes": tv_tensors.BoundingBoxes(  # pyright: ignore[reportCallIssue]
                torch.zeros(0, 4),
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=(H, W),
            ),
            "masks": tv_tensors.Mask(torch.zeros(0, H, W, dtype=torch.bool)),
            "labels": torch.zeros(0, dtype=torch.int64),
        }
        sanitize = SanitizeInstances(min_mask_area=10, labels_getter=labels_getter)
        _, out_target = sanitize(image, target)

        assert out_target["masks"].shape[0] == 0
        assert out_target["boxes"].shape[0] == 0

    def test_negative_min_mask_area_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            SanitizeInstances(min_mask_area=-1)

    def test_no_labels_getter(self) -> None:
        """Without labels_getter, boxes/masks still filter; labels ignored."""
        image, target = _build_sample(mask_areas=[100, 5])
        sanitize = SanitizeInstances(min_mask_area=10)
        _, out_target = sanitize(image, target)

        assert out_target["masks"].shape[0] == 1
        assert out_target["boxes"].shape[0] == 1
        # labels untouched (no getter)
        assert out_target["labels"].shape[0] == 2


class TestBboxFallback:
    def test_no_masks_falls_back_to_bbox(self) -> None:
        """When the input has no masks, filter on bbox extent (>0)."""
        image = tv_tensors.Image(torch.zeros(3, H, W, dtype=torch.uint8))
        target = {
            "boxes": tv_tensors.BoundingBoxes(  # pyright: ignore[reportCallIssue]
                torch.tensor(
                    [[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 5.0, 5.0]],
                    dtype=torch.float32,
                ),
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=(H, W),
            ),
            "labels": torch.tensor([1, 2], dtype=torch.int64),
        }

        def boxes_labels_getter(
            sample: tuple[tv_tensors.Image, dict[str, Any]],
        ) -> tuple[tv_tensors.BoundingBoxes, torch.Tensor]:
            return (sample[1]["boxes"], sample[1]["labels"])

        sanitize = SanitizeInstances(labels_getter=boxes_labels_getter)
        _, out_target = sanitize(image, target)

        assert out_target["boxes"].shape[0] == 1
        assert out_target["labels"].tolist() == [1]
