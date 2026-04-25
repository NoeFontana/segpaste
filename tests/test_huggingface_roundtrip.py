"""Round-trip: ``DenseSample -> HF dict -> DenseSample`` preserves invariants."""

from __future__ import annotations

import torch

from segpaste._internal.invariants.panoptic import (
    assert_panoptic_pixel_bijection,
    assert_panoptic_thing_stuff_consistent,
)
from segpaste.integrations.huggingface import from_hf_format, to_hf_format
from tests.shared import make_disjoint_panoptic_sample, make_thing_stuff_schema


def test_hf_shape() -> None:
    schema = make_thing_stuff_schema()
    sample = make_disjoint_panoptic_sample(0)
    hf = to_hf_format(sample, schema)
    assert set(hf.keys()) >= {"mask_labels", "class_labels", "pixel_values"}
    assert hf["mask_labels"].shape == (2, 24, 24)
    assert hf["mask_labels"].dtype == torch.bool
    assert hf["class_labels"].shape == (2,)
    assert hf["class_labels"].dtype == torch.int64
    assert hf["pixel_values"].shape == (3, 24, 24)


def test_round_trip_preserves_masks_and_labels() -> None:
    schema = make_thing_stuff_schema()
    for seed in range(50):
        sample = make_disjoint_panoptic_sample(seed)
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
    schema = make_thing_stuff_schema()
    for seed in range(20):
        sample = make_disjoint_panoptic_sample(seed)
        reconstructed = from_hf_format(to_hf_format(sample, schema), schema)
        assert_panoptic_thing_stuff_consistent(reconstructed, schema)
        assert_panoptic_pixel_bijection(reconstructed)


def test_round_trip_reconstructed_semantic_matches() -> None:
    """``from_hf_format`` rebuilds ``semantic_map`` from masks + labels."""
    schema = make_thing_stuff_schema()
    sample = make_disjoint_panoptic_sample(7)
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
