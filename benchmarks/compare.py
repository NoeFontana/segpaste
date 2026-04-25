"""Regression gate for the benchmark harness.

Two modes:

* ``regression`` (default) — compares a fresh JSON report against the
  committed baseline and exits non-zero when the current median exceeds
  ``baseline * (1 + threshold)``. Used by ``bench.yml`` on the CPU lane.
* ``speedup-vs-cpu-baseline`` — compares a GPU report against the CPU
  baseline and exits non-zero when the GPU median is not at least
  ``min_speedup x`` faster than the CPU median. Used by ``bench-gpu.yml``
  on the manual-dispatch A100 run (ADR-0002 Part v, ADR-0008 §v).
"""

import argparse
import json
import sys
from typing import Any


def _load(path: str) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _shape_mismatch(base: dict[str, Any], cur: dict[str, Any]) -> bool:
    return base.get("batch_size") != cur.get("batch_size") or base.get(
        "image_size"
    ) != cur.get("image_size")


def _mode_regression(
    base: dict[str, Any], cur: dict[str, Any], threshold: float
) -> int:
    if base.get("device") != cur.get("device"):
        print(
            f"device mismatch: baseline={base.get('device')} "
            f"current={cur.get('device')}",
            file=sys.stderr,
        )
        return 2

    if _shape_mismatch(base, cur):
        print(
            "workload shape mismatch (batch_size / image_size) - "
            "refresh the baseline via a PR labeled `skip-perf-gate`.",
            file=sys.stderr,
        )
        return 2

    base_median = base["median_ns"]
    cur_median = cur["median_ns"]
    delta = (cur_median - base_median) / base_median

    print(
        f"baseline median {base_median / 1e6:.2f} ms  "
        f"current median {cur_median / 1e6:.2f} ms  "
        f"delta {delta:+.2%}  threshold {threshold:+.2%}"
    )

    if delta > threshold:
        print(
            f"REGRESSION: +{delta:.2%} exceeds threshold {threshold:.2%}",
            file=sys.stderr,
        )
        return 1
    return 0


def _mode_speedup(base: dict[str, Any], cur: dict[str, Any], min_speedup: float) -> int:
    """Gate the GPU report on ``>= min_speedup`` vs. the CPU baseline."""
    if "median_ns" not in base or "median_ns" not in cur:
        print(
            "speedup-vs-cpu-baseline: report missing median_ns "
            "(GPU baseline placeholder - dispatch bench-gpu.yml to populate).",
            file=sys.stderr,
        )
        return 2
    if base.get("device") != "cpu":
        print(
            f"expected CPU baseline, got device={base.get('device')!r}",
            file=sys.stderr,
        )
        return 2
    if cur.get("device") != "cuda":
        print(
            f"expected CUDA current, got device={cur.get('device')!r}",
            file=sys.stderr,
        )
        return 2
    if _shape_mismatch(base, cur):
        print(
            "workload shape mismatch (batch_size / image_size); GPU bench "
            "must use the same workload shape as the CPU baseline.",
            file=sys.stderr,
        )
        return 2

    base_median = base["median_ns"]
    cur_median = cur["median_ns"]
    speedup = base_median / cur_median

    print(
        f"cpu baseline median {base_median / 1e6:.2f} ms  "
        f"gpu current median {cur_median / 1e6:.2f} ms  "
        f"speedup {speedup:.2f}x  min_speedup {min_speedup:.2f}x"
    )

    if speedup < min_speedup:
        print(
            f"INSUFFICIENT SPEEDUP: {speedup:.2f}x < required {min_speedup:.2f}x",
            file=sys.stderr,
        )
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("regression", "speedup-vs-cpu-baseline"),
        default="regression",
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--min-speedup", type=float, default=2.0)
    args = parser.parse_args(argv)

    base = _load(args.baseline)
    cur = _load(args.current)

    if args.mode == "regression":
        return _mode_regression(base, cur, args.threshold)
    return _mode_speedup(base, cur, args.min_speedup)


if __name__ == "__main__":
    sys.exit(main())
