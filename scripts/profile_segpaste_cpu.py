"""Profile segpaste on the worst CPU cell (debug tool; not a benchmark).

Wall-clock numbers belong in :mod:`benchmarks.comparison.sweep`. This
script is a flamegraph-shaped view of which operators dominate the hot
path; not stable enough for quantitative comparison across runs.
"""

from __future__ import annotations

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from benchmarks.comparison.impls._base import build
from benchmarks.comparison.workload import Workload


def main() -> int:
    workload = Workload(
        batch_size=8,
        image_size=512,
        k_range=(1, 20),
        max_instances=32,
        seed=0,
        n_batches=4,
        device=torch.device("cpu"),
    )
    impl = build("segpaste")
    batches = impl.adapt(workload.build_batches())

    for _ in range(5):
        impl.step(batches[0])

    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        for i in range(50):
            with record_function(f"step_{i}"):
                impl.step(batches[i % len(batches)])

    for title, sort_key in (
        ("Top 25 CPU ops by self_cpu_time_total", "self_cpu_time_total"),
        ("Top 25 CPU ops by cpu_time_total (includes child time)", "cpu_time_total"),
    ):
        print(f"\n== {title} ==")
        print(
            prof.key_averages().table(
                sort_by=sort_key, row_limit=25, top_level_events_only=False
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
