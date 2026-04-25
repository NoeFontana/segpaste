"""Smoke test for `scripts/visualize_preset.py` (ADR-0009 §5)."""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path
from types import ModuleType

import torch

from segpaste._internal.viz.manifest import (
    DatasetManifest,
    InvariantLog,
    RunManifest,
)
from segpaste._internal.viz.pipeline import run_preset
from segpaste._internal.viz.synthetic import make_synthetic_samples
from segpaste._internal.viz.writer import write_gallery
from segpaste.augmentation.batch_copy_paste import BatchCopyPasteConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "visualize_preset.py"
NUM_SAMPLES = 8
SEED = 0xC0FFEE
WALL_CLOCK_BUDGET_S = 30.0


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("visualize_preset", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_happy_path_renders_full_gallery_under_30s(tmp_path: Path) -> None:
    out_dir = tmp_path / "gallery"
    cli = _load_script()

    started = time.monotonic()
    code = cli.main(
        [
            "--num-samples",
            str(NUM_SAMPLES),
            "--seed",
            str(SEED),
            "--out-dir",
            str(out_dir),
            "--device",
            "cpu",
        ]
    )
    elapsed = time.monotonic() - started

    assert code == 0
    assert elapsed < WALL_CLOCK_BUDGET_S, f"viz took {elapsed:.2f}s (budget 30s)"

    assert (out_dir / "contact_sheet.png").is_file()
    samples_dir = out_dir / "samples"
    for i in range(NUM_SAMPLES):
        for view in ("orig", "aug", "overlay"):
            assert (samples_dir / f"{i:04d}_{view}.png").is_file()

    log = InvariantLog.model_validate_json((out_dir / "invariant_log.json").read_text())
    assert len(log.samples) == NUM_SAMPLES
    assert all(r.ok for s in log.samples for r in s.reports)

    dataset = DatasetManifest.model_validate_json(
        (out_dir / "dataset_manifest.json").read_text()
    )
    assert dataset.source == "synthetic"
    assert dataset.sample_count == NUM_SAMPLES

    run = RunManifest.model_validate_json((out_dir / "run_manifest.json").read_text())
    assert run.seed == SEED
    assert run.num_samples == NUM_SAMPLES

    failed_dir = out_dir / "_failed"
    assert not failed_dir.exists() or not any(failed_dir.iterdir())


def test_force_overlap_populates_failed_tree(tmp_path: Path) -> None:
    out_dir = tmp_path / "gallery"
    samples = make_synthetic_samples(seed=SEED, count=NUM_SAMPLES)
    outcomes = run_preset(
        BatchCopyPasteConfig(),
        samples,
        seed=SEED,
        device=torch.device("cpu"),
        force_overlap=True,
    )

    assert any(not r.ok for o in outcomes for r in o.reports)
    assert any(
        r.name == "instance.no_same_class_overlap" and not r.ok
        for o in outcomes
        for r in o.reports
    )

    all_ok = write_gallery(
        out_dir,
        outcomes,
        preset="default",
        seed=SEED,
        batch_size=NUM_SAMPLES,
        device="cpu",
    )
    assert not all_ok
    failed_dir = out_dir / "_failed"
    assert failed_dir.is_dir()
    assert any(failed_dir.iterdir())
