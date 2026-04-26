"""Synthetic invariant tests for the ``coco-panoptic`` preset (ADR-0009 §5)."""

from __future__ import annotations

import torch

from segpaste._internal.invariants.panoptic import (
    assert_panoptic_pixel_bijection,
    assert_panoptic_thing_stuff_consistent,
)
from segpaste.augmentation.batch_copy_paste import BatchCopyPaste
from segpaste.presets import get_preset
from segpaste.types import BatchedDenseSample, DenseSample
from tests.fixtures.loader import load_fixture
from tests.shared import make_thing_stuff_schema


def test_coco_panoptic_preset_round_trip() -> None:
    cfg = get_preset("coco-panoptic")
    assert cfg.name == "coco-panoptic"
    assert cfg.batch_copy_paste.placement.paste_prob == 0.5
    assert cfg.batch_copy_paste.placement.k_range == (1, 20)
    assert cfg.batch_copy_paste.panoptic is not None
    assert cfg.batch_copy_paste.panoptic.tau_stuff_frac == 0.1
    classes = cfg.batch_copy_paste.panoptic.taxonomy.classes
    assert sum(1 for v in classes.values() if v == "thing") == 80
    assert sum(1 for v in classes.values() if v == "stuff") == 53


def test_coco_panoptic_invariants_pass_after_paste() -> None:
    cfg = get_preset("coco-panoptic")
    panoptic_cfg = cfg.batch_copy_paste.panoptic
    assert panoptic_cfg is not None
    # Substitute the synthetic fixture's 3-class taxonomy and pin paste_prob=1.0
    # so the composite path always fires (preset default 0.5 can yield empty pastes).
    schema = make_thing_stuff_schema(max_instances_per_image=4)
    pinned = cfg.batch_copy_paste.model_copy(
        update={
            "placement": cfg.batch_copy_paste.placement.model_copy(
                update={"paste_prob": 1.0}
            ),
            "panoptic": panoptic_cfg.model_copy(update={"taxonomy": schema}),
        }
    )
    base = load_fixture("panoptic_stuff_and_things")
    padded = BatchedDenseSample.from_samples([base, base, base, base]).to_padded(
        max_instances=4
    )
    mod = BatchCopyPaste(pinned)
    out = mod(padded, torch.Generator().manual_seed(0))
    samples: list[DenseSample] = BatchedDenseSample.from_padded(out).to_samples()
    for sample in samples:
        assert_panoptic_thing_stuff_consistent(sample, schema)
        assert_panoptic_pixel_bijection(sample)
