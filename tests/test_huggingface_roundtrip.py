"""Round-trip: ``DenseSample -> HF dict -> DenseSample`` preserves invariants."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import torch
from torchvision import tv_tensors

from segpaste.integrations.huggingface import from_hf_format, to_hf_format
from segpaste.types import (
    DenseSample,
    InstanceMask,
    PanopticMap,
    PanopticSchema,
    SemanticMap,
)
from tests.invariants.panoptic import (
    assert_panoptic_pixel_bijection,
    assert_panoptic_thing_stuff_consistent,
)


@dataclass
class _FakeSchema:
    classes: Mapping[int, Literal["thing", "stuff"]]
    ignore_index: int
    max_instances_per_image: int


def _schema() -> PanopticSchema:
    return _FakeSchema(
        classes={0: "stuff", 1: "thing", 2: "thing"},
        ignore_index=255,
        max_instances_per_image=256,
    )


def _mk_sample(seed: int) -> DenseSample:
    g = torch.Generator().manual_seed(seed)
    h, w = 24, 24
    image = tv_tensors.Image(torch.rand(3, h, w, generator=g, dtype=torch.float32))
    # One thing of class 1, one thing of class 2, disjoint
    masks = torch.zeros((2, h, w), dtype=torch.bool)
    masks[0, 2:10, 2:10] = True
    masks[1, 12:20, 12:20] = True
    labels = torch.tensor([1, 2], dtype=torch.int64)
    boxes_t = torch.tensor(
        [[2.0, 2.0, 10.0, 10.0], [12.0, 12.0, 20.0, 20.0]], dtype=torch.float32
    )
    sem = torch.zeros((h, w), dtype=torch.int64)
    sem[masks[0]] = 1
    sem[masks[1]] = 2
    pan = torch.zeros((h, w), dtype=torch.int64)
    pan[masks[0]] = 1
    pan[masks[1]] = 2
    return DenseSample(
        image=image,
        boxes=tv_tensors.BoundingBoxes(  # pyright: ignore[reportCallIssue]
            boxes_t, format=tv_tensors.BoundingBoxFormat.XYXY, canvas_size=(h, w)
        ),
        labels=labels,
        instance_ids=torch.arange(2, dtype=torch.int32),
        instance_masks=InstanceMask(masks),
        semantic_map=SemanticMap(sem),
        panoptic_map=PanopticMap(pan),
    )


def test_hf_shape() -> None:
    schema = _schema()
    sample = _mk_sample(0)
    hf = to_hf_format(sample, schema)
    assert set(hf.keys()) >= {"mask_labels", "class_labels", "pixel_values"}
    assert hf["mask_labels"].shape == (2, 24, 24)
    assert hf["mask_labels"].dtype == torch.bool
    assert hf["class_labels"].shape == (2,)
    assert hf["class_labels"].dtype == torch.int64
    assert hf["pixel_values"].shape == (3, 24, 24)


def test_round_trip_preserves_masks_and_labels() -> None:
    schema = _schema()
    for seed in range(50):
        sample = _mk_sample(seed)
        hf = to_hf_format(sample, schema)
        reconstructed = from_hf_format(hf, schema)

        assert reconstructed.instance_masks is not None
        assert reconstructed.semantic_map is not None
        assert reconstructed.panoptic_map is not None

        orig_masks = sample.instance_masks.as_subclass(torch.Tensor)  # pyright: ignore[reportOptionalMemberAccess]
        recon_masks = reconstructed.instance_masks.as_subclass(torch.Tensor)
        assert torch.equal(orig_masks, recon_masks)
        assert torch.equal(sample.labels, reconstructed.labels)


def test_round_trip_preserves_panoptic_invariants() -> None:
    schema = _schema()
    for seed in range(20):
        sample = _mk_sample(seed)
        reconstructed = from_hf_format(to_hf_format(sample, schema), schema)
        assert_panoptic_thing_stuff_consistent(reconstructed, schema)
        assert_panoptic_pixel_bijection(reconstructed)


def test_round_trip_reconstructed_semantic_matches() -> None:
    """``from_hf_format`` rebuilds ``semantic_map`` from masks + labels."""
    schema = _schema()
    sample = _mk_sample(7)
    reconstructed = from_hf_format(to_hf_format(sample, schema), schema)

    assert sample.semantic_map is not None
    assert reconstructed.semantic_map is not None
    orig_sem = sample.semantic_map.as_subclass(torch.Tensor)
    recon_sem = reconstructed.semantic_map.as_subclass(torch.Tensor)
    # Wherever the original had a real class, reconstructed must match.
    covered = orig_sem != schema.ignore_index
    # Stuff pixels in the original (class 0) are written as ignore in
    # reconstructed because `to_hf_format` doesn't encode stuff bkgrnd.
    thing_pixels = (orig_sem != 0) & covered
    assert torch.equal(recon_sem[thing_pixels], orig_sem[thing_pixels])
