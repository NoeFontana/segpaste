"""ADR-0016 §verification: smoke tests for the multi-impl harness."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from benchmarks.comparison.impls import registry
from benchmarks.comparison.impls._base import build
from benchmarks.comparison.schemas import (
    SCHEMA_NAME,
    ComparisonReport,
    ImplResult,
    derive_throughput,
    workload_to_dict,
)
from benchmarks.comparison.sweep import GRIDS, run_sweep
from benchmarks.comparison.sweep import main as sweep_main
from benchmarks.comparison.workload import Workload


def test_registry_includes_three_impls() -> None:
    names = set(registry().keys())
    assert {"segpaste", "torchvision_ref", "mmdet"}.issubset(names)


def test_workload_label_round_trip() -> None:
    w = Workload(batch_size=2, image_size=64, k_range=(1, 3), max_instances=8, seed=42)
    assert "b2_img64_k1-3_cpu" in w.label()
    batches = w.build_batches()
    assert len(batches) == w.n_batches
    assert all(len(b) == w.batch_size for b in batches)


def test_workload_is_deterministic_under_same_seed() -> None:
    w1 = Workload(batch_size=2, image_size=32, k_range=(1, 2), max_instances=4, seed=7)
    w2 = Workload(batch_size=2, image_size=32, k_range=(1, 2), max_instances=4, seed=7)
    b1 = w1.build_batches()[0][0]
    b2 = w2.build_batches()[0][0]
    assert torch.equal(b1.image, b2.image)
    assert torch.equal(b1.masks, b2.masks)


def test_segpaste_impl_adapts_and_steps() -> None:
    impl = build("segpaste")
    w = Workload(batch_size=2, image_size=32, k_range=(1, 2), max_instances=4, seed=0)
    batches = impl.adapt(w.build_batches())
    assert batches
    impl.step(batches[0])  # smoke; no assert on outputs


def test_torchvision_ref_impl_adapts_and_steps() -> None:
    impl = build("torchvision_ref")
    w = Workload(batch_size=2, image_size=32, k_range=(1, 2), max_instances=4, seed=0)
    batches = impl.adapt(w.build_batches())
    assert batches
    impl.step(batches[0])


def test_mmdet_impl_is_gated_by_import() -> None:
    """mmdet is not in [dev] / [bench]; importing surfaces the install hint."""
    pytest.importorskip(
        "mmdet",
        reason="mmdet only loaded for [bench-mmdet] dispatch; "
        "without it the sweep emits status=skipped, which is the tested behavior",
    )


def test_mmdet_supports_device_cpu_only() -> None:
    impl = build("mmdet")
    assert impl.supports_device(torch.device("cpu")) is True
    assert impl.supports_device(torch.device("cuda")) is False


def test_derive_throughput_handles_missing_report() -> None:
    ips, bps, pps = derive_throughput(None, batch_size=8, k_range=(1, 5))
    assert ips is None and bps is None and pps is None


def test_derive_throughput_basic() -> None:
    ips, bps, pps = derive_throughput(
        {"median_ns": 1_000_000_000}, batch_size=8, k_range=(1, 5)
    )
    assert ips == pytest.approx(8.0)
    assert bps == pytest.approx(1.0)
    assert pps == pytest.approx(8.0 * 3.0)


def test_comparison_report_dict_roundtrip() -> None:
    w = Workload(batch_size=2, image_size=32, k_range=(1, 2), max_instances=4)
    report = ComparisonReport(
        workload=workload_to_dict(w),
        env={"python": "x"},
        implementations=[ImplResult(name="foo", status="skipped", skip_reason="nope")],
    )
    out = report.to_dict()
    assert out["schema"] == SCHEMA_NAME
    assert out["workload"]["batch_size"] == 2
    assert out["implementations"][0]["name"] == "foo"


def test_sweep_smoke_grid_end_to_end(tmp_path: Path) -> None:
    reports = run_sweep(
        grid=GRIDS["smoke"],
        device=torch.device("cpu"),
        seed=0,
        warmup=2,
        iters=4,
    )
    assert len(reports) == 1
    impls = {r.name for r in reports[0].implementations}
    assert {"segpaste", "torchvision_ref", "mmdet"} == impls

    out = tmp_path / "smoke.json"
    out.write_text(json.dumps([r.to_dict() for r in reports]))
    reloaded = json.loads(out.read_text())
    assert reloaded[0]["schema"] == SCHEMA_NAME


def test_sweep_cli_writes_valid_json(tmp_path: Path) -> None:
    out = tmp_path / "cli.json"
    rc = sweep_main(
        [
            "--device",
            "cpu",
            "--grid",
            "smoke",
            "--warmup",
            "2",
            "--iters",
            "4",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    data = json.loads(out.read_text())
    assert isinstance(data, list) and data
    assert data[0]["schema"] == SCHEMA_NAME
    assert data[0]["workload"]["device"] == "cpu"


def test_aggregate_renders_markdown() -> None:
    from benchmarks.comparison.aggregate import render

    reports = run_sweep(
        grid=GRIDS["smoke"],
        device=torch.device("cpu"),
        seed=0,
        warmup=2,
        iters=4,
    )
    md = render([r.to_dict() for r in reports])
    assert "# Throughput comparison (ADR-0016)" in md
    assert "B=2 · 256² · k=[1, 3]" in md
    assert "segpaste" in md
