"""CLI harness for measuring `CopyPasteCollator` throughput.

Contract pinned in ADR-0002. Emits a JSON report with per-iter median, IQR,
and the metadata needed to compare runs across machines and git revisions.
"""

import argparse
import datetime
import itertools
import json
import os
import platform
import sys
from importlib import metadata
from typing import Any

import numpy as np
import torch
import torch.utils.benchmark as tbench

from benchmarks._fixture import build_batch
from segpaste.augmentation import CopyPasteAugmentation, CopyPasteCollator
from segpaste.config import CopyPasteConfig

SCHEMA_VERSION = 1
BENCHMARK_LABEL = "CopyPasteCollator"


def _get_segpaste_version() -> str:
    try:
        return metadata.version("segpaste")
    except metadata.PackageNotFoundError:
        return "unknown"


def _get_cpu_model() -> str | None:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def _run(
    device: str,
    warmup: int,
    iters: int,
    batch_size: int,
    image_size: int,
    k_range: tuple[int, int],
    num_batches: int,
) -> dict[str, Any]:
    cfg = CopyPasteConfig(
        paste_probability=1.0,
        min_paste_objects=k_range[0],
        max_paste_objects=k_range[1],
        min_object_area=1,
    )
    aug = CopyPasteAugmentation(cfg)
    collator = CopyPasteCollator(aug)

    batches = [
        build_batch(seed=i, batch_size=batch_size, img_size=image_size, k_range=k_range)
        for i in range(num_batches)
    ]
    counter = itertools.count()

    def stmt() -> None:
        _ = collator(batches[next(counter) % num_batches])

    timer = tbench.Timer(
        stmt="stmt()",
        globals={"stmt": stmt},
        num_threads=1,
        label=BENCHMARK_LABEL,
        sub_label=f"b{batch_size}_{image_size}_{device}",
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
        "label": BENCHMARK_LABEL,
        "device": device,
        "cpu_model": _get_cpu_model(),
        "batch_size": batch_size,
        "image_size": image_size,
        "k_range": list(k_range),
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
        "segpaste_version": _get_segpaste_version(),
        "commit_sha": os.environ.get("GITHUB_SHA") or _git_short_sha(),
        "runner": os.environ.get("RUNNER_NAME") or platform.node(),
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
    }


def _git_short_sha() -> str | None:
    import subprocess

    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--k-min", type=int, default=1)
    parser.add_argument("--k-max", type=int, default=5)
    parser.add_argument("--num-batches", type=int, default=64)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args(argv)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available", file=sys.stderr)
        return 2

    report = _run(
        device=args.device,
        warmup=args.warmup,
        iters=args.iters,
        batch_size=args.batch_size,
        image_size=args.image_size,
        k_range=(args.k_min, args.k_max),
        num_batches=args.num_batches,
    )

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    print(
        f"median {report['median_ns'] / 1e6:.2f} ms  "
        f"IQR/median {report['iqr_over_median']:.2%}  "
        f"iters {report['iters']}  device {report['device']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
