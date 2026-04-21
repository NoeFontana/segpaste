"""Regression gate for the benchmark harness.

Compares a fresh JSON report against the committed baseline and exits
non-zero when the current median exceeds `baseline * (1 + threshold)`.
"""

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--threshold", type=float, default=0.05)
    args = parser.parse_args(argv)

    with open(args.baseline) as f:
        base = json.load(f)
    with open(args.current) as f:
        cur = json.load(f)

    if base.get("device") != cur.get("device"):
        print(
            f"device mismatch: baseline={base.get('device')} "
            f"current={cur.get('device')}",
            file=sys.stderr,
        )
        return 2

    if base.get("batch_size") != cur.get("batch_size") or base.get(
        "image_size"
    ) != cur.get("image_size"):
        print(
            "workload shape mismatch (batch_size / image_size) — "
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
        f"delta {delta:+.2%}  threshold {args.threshold:+.2%}"
    )

    if delta > args.threshold:
        print(
            f"REGRESSION: +{delta:.2%} exceeds threshold {args.threshold:.2%}",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
