"""Profile segpaste on the worst CPU cell."""

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

    # Warmup
    for _ in range(5):
        impl.step(batches[0])

    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        for i in range(10):
            with record_function(f"step_{i}"):
                impl.step(batches[i % len(batches)])

    print("\n== Top 25 CPU ops by self_cpu_time_total ==")
    print(
        prof.key_averages().table(
            sort_by="self_cpu_time_total", row_limit=25, top_level_events_only=False
        )
    )

    print("\n== Top 25 CPU ops by cpu_time_total (includes child time) ==")
    print(
        prof.key_averages().table(
            sort_by="cpu_time_total", row_limit=25, top_level_events_only=False
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
