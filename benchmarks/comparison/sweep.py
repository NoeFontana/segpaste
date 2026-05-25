"""CLI driver for the 3D throughput sweep (ADR-0016 §3).

Default grid:
``batch in {2, 8, 32} x image in {256, 512, 1024} x k_hi in {5, 20}``
= 18 workloads. Add ``--grid smoke`` for the 1-cell smoke run used by
tests and the ADR's verification block.

For each workload, every registered implementation runs through
:func:`benchmarks._harness.run_benchmark` (same Timer, warmup, iters
contract as ADR-0002). Adapters run outside the timed window; mmdet's
NotInstalled error becomes a ``status: "skipped"`` entry rather than a
sweep-wide failure.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import torch

from benchmarks._harness import batch_stmt, run_benchmark
from benchmarks.comparison.impls import registry
from benchmarks.comparison.impls._base import Implementation, build
from benchmarks.comparison.impls.mmdet_copypaste import MmdetNotInstalledError
from benchmarks.comparison.schemas import (
    ComparisonReport,
    ImplResult,
    collect_env,
    derive_throughput,
    workload_to_dict,
)
from benchmarks.comparison.workload import Workload

logger = logging.getLogger("benchmarks.comparison.sweep")


@dataclass(frozen=True, slots=True)
class GridSpec:
    batch_sizes: tuple[int, ...]
    image_sizes: tuple[int, ...]
    k_his: tuple[int, ...]


GRIDS: dict[str, GridSpec] = {
    # CPU-friendly: 12 cells, ~15-20 min per run on a fast desktop. Excludes
    # 1024^2 (CPU compositor cost grows linearly with HxWxB; 1024^2 is the
    # GPU sweep's territory).
    "default": GridSpec(
        batch_sizes=(2, 8, 32),
        image_sizes=(256, 512),
        k_his=(5, 20),
    ),
    # Full 3D grid; assumes a GPU. 18 cells. With CUDA at 1024^2 each cell
    # is sub-second, so the full pass runs in a few minutes.
    "full": GridSpec(
        batch_sizes=(2, 8, 32),
        image_sizes=(256, 512, 1024),
        k_his=(5, 20),
    ),
    "smoke": GridSpec(
        batch_sizes=(2,),
        image_sizes=(256,),
        k_his=(3,),
    ),
    "anchor": GridSpec(
        batch_sizes=(8,),
        image_sizes=(512,),
        k_his=(5,),
    ),
}


def _iter_workloads(
    grid: GridSpec, device: torch.device, seed: int
) -> Iterable[Workload]:
    for batch_size, image_size, k_hi in product(
        grid.batch_sizes, grid.image_sizes, grid.k_his
    ):
        yield Workload(
            batch_size=batch_size,
            image_size=image_size,
            k_range=(1, k_hi),
            max_instances=max(k_hi + 1, 32),
            seed=seed,
            n_batches=8,
            device=device,
        )


def _run_one_impl(
    impl: Implementation,
    workload: Workload,
    *,
    warmup: int,
    iters: int,
) -> ImplResult:
    if not impl.supports_device(workload.device):
        return ImplResult(
            name=impl.name,
            status="skipped",
            skip_reason=f"impl does not support device {workload.device.type!r}",
        )

    try:
        batches = impl.adapt(workload.build_batches())
    except MmdetNotInstalledError as exc:
        return ImplResult(name=impl.name, status="skipped", skip_reason=str(exc))
    except Exception as exc:  # surfacing the error is the point
        logger.exception("adapt() failed for %s", impl.name)
        return ImplResult(name=impl.name, status="error", error_message=repr(exc))

    stmt = batch_stmt(list(batches), impl.step)
    try:
        report = run_benchmark(
            label=impl.name,
            device=workload.device.type,
            warmup=warmup,
            iters=iters,
            sub_label=workload.label(),
            stmt=stmt,
            extra_fields={
                "batch_size": workload.batch_size,
                "image_size": workload.image_size,
                "k_range": list(workload.k_range),
                "max_instances": workload.max_instances,
                "compile": False,
                "peak_memory_bytes": 0,
            },
        )
    except Exception as exc:
        logger.exception("step() failed for %s", impl.name)
        return ImplResult(name=impl.name, status="error", error_message=repr(exc))

    images_per_sec, batches_per_sec, pasted_per_sec = derive_throughput(
        report, workload.batch_size, workload.k_range
    )
    return ImplResult(
        name=impl.name,
        status="ok",
        report=report,
        images_per_sec=images_per_sec,
        batches_per_sec=batches_per_sec,
        pasted_instances_per_sec=pasted_per_sec,
    )


def run_sweep(
    *,
    grid: GridSpec,
    device: torch.device,
    seed: int,
    warmup: int,
    iters: int,
    only: tuple[str, ...] | None = None,
) -> list[ComparisonReport]:
    impl_names = tuple(registry().keys())
    if only is not None:
        unknown = set(only) - set(impl_names)
        if unknown:
            raise ValueError(f"unknown implementations: {sorted(unknown)}")
        impl_names = only

    reports: list[ComparisonReport] = []
    env = collect_env()

    for workload in _iter_workloads(grid, device, seed):
        report = ComparisonReport(
            workload=workload_to_dict(workload),
            env=env,
        )
        for name in impl_names:
            impl = build(name)
            logger.info("running %s @ %s", name, workload.label())
            report.implementations.append(
                _run_one_impl(impl, workload, warmup=warmup, iters=iters)
            )
        reports.append(report)
    return reports


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--grid", choices=sorted(GRIDS.keys()), default="default")
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="restrict to a subset of registered implementations",
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="info-level progress logging"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable", file=sys.stderr)
        return 2

    device = torch.device(args.device)
    grid = GRIDS[args.grid]
    reports = run_sweep(
        grid=grid,
        device=device,
        seed=args.seed,
        warmup=args.warmup,
        iters=args.iters,
        only=tuple(args.only) if args.only else None,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps([r.to_dict() for r in reports], indent=2))
    print(
        f"wrote {args.out}  workloads={len(reports)}  "
        f"impls={[i.name for i in (reports[0].implementations if reports else [])]}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
