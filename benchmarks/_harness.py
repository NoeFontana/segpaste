"""Shared bench harness: build the `timeit` loop, gather metadata, emit JSON.

Report schema (``schema_version=1``) is pinned in ADR-0002 for the
CPU baseline and ADR-0008 Part v for the GPU throughput lane.
``compare.py`` gates any bench that emits this schema.
"""

from __future__ import annotations

import datetime
import itertools
import os
import platform
import subprocess
from collections.abc import Callable
from importlib import metadata
from typing import Any

import numpy as np
import torch
import torch.utils.benchmark as tbench

SCHEMA_VERSION = 1


def _segpaste_version() -> str:
    try:
        return metadata.version("segpaste")
    except metadata.PackageNotFoundError:
        return "unknown"


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


def run_benchmark(
    *,
    label: str,
    device: str,
    warmup: int,
    iters: int,
    sub_label: str,
    stmt: Callable[[], None],
    extra_fields: dict[str, Any],
) -> dict[str, Any]:
    """Run ``stmt`` through ``torch.utils.benchmark.Timer`` and build a report.

    ``extra_fields`` is merged verbatim into the output dict so per-bench
    workload shape (batch size, image size, k_range, etc.) can be recorded
    alongside the shared schema.
    """
    timer = tbench.Timer(
        stmt="stmt()",
        globals={"stmt": stmt},
        num_threads=1,
        label=label,
        sub_label=sub_label,
    )

    for _ in range(warmup):
        stmt()
    if device == "cuda":
        torch.cuda.synchronize()

    samples_ns = np.array(
        [timer.timeit(1).median * 1e9 for _ in range(iters)], dtype=np.float64
    )
    q1, median, q3 = np.percentile(samples_ns, [25, 50, 75])
    iqr = q3 - q1

    return {
        "schema_version": SCHEMA_VERSION,
        "label": label,
        "device": device,
        "cpu_model": _cpu_model(),
        **extra_fields,
        "warmup": warmup,
        "iters": iters,
        "median_ns": int(median),
        "q1_ns": int(q1),
        "q3_ns": int(q3),
        "iqr_ns": int(iqr),
        "iqr_over_median": float(iqr / median) if median > 0 else float("nan"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": metadata.version("torchvision"),
        "segpaste_version": _segpaste_version(),
        "commit_sha": os.environ.get("GITHUB_SHA") or _git_short_sha(),
        "runner": os.environ.get("RUNNER_NAME") or platform.node(),
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
    }


def batch_stmt(batches: list[Any], consume: Callable[[Any], Any]) -> Callable[[], None]:
    """Build a 0-arg ``stmt()`` that feeds ``batches`` round-robin into ``consume``."""
    counter = itertools.count()

    def stmt() -> None:
        consume(batches[next(counter) % len(batches)])

    return stmt
