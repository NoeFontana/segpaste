"""Synthetic invariant tests for the ``coco-instance`` preset (ADR-0009 §5)."""

from __future__ import annotations

import torch

from segpaste._internal.invariants.instance import (
    assert_instance_no_same_class_overlap,
)
from segpaste.augmentation.batch_copy_paste import BatchCopyPaste
from segpaste.presets import get_preset
from segpaste.types import BatchedDenseSample
from tests.fixtures.loader import load_fixture


def test_coco_instance_preset_round_trip() -> None:
    cfg = get_preset("coco-instance")
    assert cfg.name == "coco-instance"
    assert cfg.batch_copy_paste.placement.k_range == (1, 32)
    assert cfg.batch_copy_paste.min_residual_area_frac == 0.1
    assert cfg.batch_copy_paste.panoptic is None


def test_coco_instance_invariants_pass_after_paste() -> None:
    cfg = get_preset("coco-instance")
    base = load_fixture("two_overlapping_things")
    padded = BatchedDenseSample.from_samples([base, base, base, base]).to_padded(
        max_instances=4
    )
    mod = BatchCopyPaste(cfg.batch_copy_paste)
    out = mod(padded, torch.Generator().manual_seed(0))
    samples = BatchedDenseSample.from_padded(out).to_samples()
    for sample in samples:
        assert_instance_no_same_class_overlap(sample)
