"""Integration test for the FiftyOne export (P4)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import torch

pytest.importorskip("fiftyone")

from segpaste._internal.viz.fiftyone_export import build_dataset
from segpaste._internal.viz.pipeline import run_preset
from segpaste._internal.viz.synthetic import make_synthetic_samples
from segpaste._internal.viz.writer import write_gallery
from segpaste.augmentation.batch_copy_paste import BatchCopyPasteConfig

NUM_SAMPLES = 4
SEED = 0xC0FFEE


def test_build_dataset_populates_documented_fields(tmp_path: Path) -> None:
    out_dir = tmp_path / "gallery"
    samples = make_synthetic_samples(seed=SEED, count=NUM_SAMPLES)
    outcomes = run_preset(
        BatchCopyPasteConfig(),
        samples,
        seed=SEED,
        device=torch.device("cpu"),
    )
    write_gallery(
        out_dir,
        outcomes,
        preset="default",
        seed=SEED,
        batch_size=NUM_SAMPLES,
        device="cpu",
    )

    dataset = build_dataset(
        out_dir=out_dir,
        outcomes=outcomes,
        name="segpaste-test-happy",
        info={"preset": "default", "seed": SEED},
    )

    try:
        import fiftyone as fo

        assert dataset.count() == NUM_SAMPLES
        info = cast(dict[str, Any], dataset.info)
        assert info["preset"] == "default"
        assert info["seed"] == SEED

        seen_indices: set[int] = set()
        for fo_sample in dataset:
            assert Path(fo_sample.filepath).is_file()
            assert Path(fo_sample.original_filepath).is_file()
            assert Path(fo_sample.overlay_filepath).is_file()
            assert isinstance(fo_sample.invariant_passed, bool)
            assert fo_sample.invariant_passed is True
            assert isinstance(fo_sample.failed_checks, list)
            assert fo_sample.failed_checks == []
            assert isinstance(fo_sample.K_pasted, int)
            assert fo_sample.K_pasted >= 0
            assert isinstance(fo_sample.paste_area_frac, float)
            assert 0.0 <= fo_sample.paste_area_frac <= 1.0
            assert isinstance(fo_sample.detections, fo.Detections)
            assert isinstance(fo_sample.original_detections, fo.Detections)
            detections = cast(list[fo.Detection], fo_sample.detections.detections)
            for det in detections:
                assert isinstance(det, fo.Detection)
                assert isinstance(det.label, str)
                bbox = cast(list[float], det.bounding_box)
                assert len(bbox) == 4
                assert all(0.0 <= float(v) <= 1.0 for v in bbox)
                mask = cast("Any", det.mask)
                assert mask is not None
                assert mask.ndim == 2
                assert mask.shape[0] > 0 and mask.shape[1] > 0
            seen_indices.add(int(fo_sample.sample_index))
        assert seen_indices == set(range(NUM_SAMPLES))
    finally:
        dataset.delete()


def test_failed_checks_lists_failed_invariant_names(tmp_path: Path) -> None:
    out_dir = tmp_path / "gallery"
    samples = make_synthetic_samples(seed=SEED, count=NUM_SAMPLES)
    outcomes = run_preset(
        BatchCopyPasteConfig(),
        samples,
        seed=SEED,
        device=torch.device("cpu"),
        force_overlap=True,
    )
    write_gallery(
        out_dir,
        outcomes,
        preset="default",
        seed=SEED,
        batch_size=NUM_SAMPLES,
        device="cpu",
    )

    dataset = build_dataset(
        out_dir=out_dir,
        outcomes=outcomes,
        name="segpaste-test-failed",
    )

    try:
        any_failed = False
        for fo_sample in dataset:
            if not fo_sample.invariant_passed:
                any_failed = True
                assert "instance.no_same_class_overlap" in fo_sample.failed_checks
        assert any_failed
    finally:
        dataset.delete()
