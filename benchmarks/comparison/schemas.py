"""``comparison_v1`` JSON schema (ADR-0016 §5).

A single workload produces one :class:`ComparisonReport`. Each
implementation under that workload contributes an :class:`ImplResult`.
The ADR-0002 schema v1 single-impl ``report`` is embedded verbatim per
implementation so downstream tooling can pull a single entry and feed
it to existing ``benchmarks.compare`` without conversion.
"""

from __future__ import annotations

import datetime
import os
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from importlib import metadata
from typing import Any, Literal

import torch

from benchmarks.comparison.workload import Workload

SCHEMA_NAME = "comparison_v1"


@dataclass(slots=True)
class ImplResult:
    """One implementation's contribution to a comparison report."""

    name: str
    status: Literal["ok", "skipped", "error"]
    report: dict[str, Any] | None = None
    skip_reason: str | None = None
    error_message: str | None = None
    images_per_sec: float | None = None
    batches_per_sec: float | None = None
    pasted_instances_per_sec: float | None = None


@dataclass(slots=True)
class ComparisonReport:
    """Outer envelope: one workload, list of implementation results."""

    schema_version: int = 1
    schema: str = SCHEMA_NAME
    workload: dict[str, Any] = field(default_factory=dict)
    env: dict[str, Any] = field(default_factory=dict)
    implementations: list[ImplResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "schema": self.schema,
            "workload": self.workload,
            "env": self.env,
            "implementations": [asdict(r) for r in self.implementations],
        }


def workload_to_dict(workload: Workload) -> dict[str, Any]:
    return {
        "batch_size": workload.batch_size,
        "image_size": workload.image_size,
        "k_range": list(workload.k_range),
        "max_instances": workload.max_instances,
        "seed": workload.seed,
        "device": workload.device.type,
        "n_batches": workload.n_batches,
    }


def derive_throughput(
    report: dict[str, Any] | None, batch_size: int, k_range: tuple[int, int]
) -> tuple[float | None, float | None, float | None]:
    """Return (images_per_sec, batches_per_sec, pasted_instances_per_sec).

    Uses the average of ``k_range`` endpoints as the expected per-sample
    paste count. Returns ``(None, None, None)`` when no median is available.
    """
    if report is None or "median_ns" not in report or report["median_ns"] <= 0:
        return None, None, None
    median_s = report["median_ns"] / 1e9
    batches_per_sec = 1.0 / median_s
    images_per_sec = batch_size * batches_per_sec
    avg_k = 0.5 * (k_range[0] + k_range[1])
    pasted_per_sec = images_per_sec * avg_k
    return images_per_sec, batches_per_sec, pasted_per_sec


def collect_env() -> dict[str, Any]:
    """Capture the runtime environment alongside the bench numbers."""
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": _safe_version("torchvision"),
        "numpy": _safe_version("numpy"),
        "mmdet": _safe_version("mmdet"),
        "segpaste_version": _safe_version("segpaste"),
        "cpu_model": _cpu_model(),
        "runner": os.environ.get("RUNNER_NAME") or platform.node(),
        "commit_sha": os.environ.get("GITHUB_SHA") or _git_short_sha(),
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
    }


def _safe_version(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def _cpu_model() -> str | None:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def _git_short_sha() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


__all__ = [
    "SCHEMA_NAME",
    "ComparisonReport",
    "ImplResult",
    "collect_env",
    "derive_throughput",
    "workload_to_dict",
]
