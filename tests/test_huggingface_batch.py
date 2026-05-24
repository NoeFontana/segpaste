"""ADR-0015 §1: HF batch verbs — ``to_hf_batch`` + ``make_hf_collate_fn``."""

from __future__ import annotations

import torch

from segpaste import (
    BatchedDenseSample,
    make_hf_collate_fn,
    to_hf_batch,
)
from tests.shared import make_disjoint_panoptic_sample


def test_to_hf_batch_shape() -> None:
    samples = [make_disjoint_panoptic_sample(i) for i in range(4)]
    padded = BatchedDenseSample.from_samples(samples).to_padded(max_instances=8)

    hf = to_hf_batch(padded)

    assert set(hf.keys()) >= {"mask_labels", "class_labels", "pixel_values"}

    pixel_values = hf["pixel_values"]
    assert isinstance(pixel_values, torch.Tensor)
    assert pixel_values.shape == (4, 3, 24, 24)
    assert pixel_values.dtype == torch.float32

    mask_labels = hf["mask_labels"]
    class_labels = hf["class_labels"]
    assert isinstance(mask_labels, list)
    assert isinstance(class_labels, list)
    assert len(mask_labels) == 4
    assert len(class_labels) == 4
    for masks_i, labels_i in zip(mask_labels, class_labels, strict=True):
        assert masks_i.shape == (2, 24, 24)
        assert masks_i.dtype == torch.bool
        assert labels_i.shape == (2,)
        assert labels_i.dtype == torch.int64


def test_make_hf_collate_fn_closes_loop() -> None:
    collate = make_hf_collate_fn("coco-panoptic", max_instances=8)
    samples = [make_disjoint_panoptic_sample(i) for i in range(2)]

    hf = collate(samples)

    pixel_values = hf["pixel_values"]
    assert isinstance(pixel_values, torch.Tensor)
    assert pixel_values.shape == (2, 3, 24, 24)
    mask_labels = hf["mask_labels"]
    assert isinstance(mask_labels, list)
    assert len(mask_labels) == 2
