"""Throughput bench for :class:`BatchCopyPaste` (ADR-0002 Part v, ADR-0008).

CPU lane: runs on PRs via ``bench.yml``; reads ``benchmarks/baseline.json``.
GPU lane: ``workflow_dispatch`` only via ``bench-gpu.yml``; emits a report
shaped like ``baseline.json`` plus ``peak_memory_bytes`` so
``compare.py --mode speedup-vs-cpu-baseline`` can gate on >=2x speedup.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from benchmarks._harness import batch_stmt, run_benchmark
from benchmarks._padded_fixture import build_batches
from segpaste import BatchCopyPaste


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--k-lo", type=int, default=1)
    parser.add_argument("--k-hi", type=int, default=5)
    parser.add_argument("--max-instances", type=int, default=5)
    parser.add_argument("--n-batches", type=int, default=8)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable", file=sys.stderr)
        return 2

    device = torch.device(args.device)
    batches = build_batches(
        n_batches=args.n_batches,
        batch_size=args.batch_size,
        image_size=args.image_size,
        k_range=(args.k_lo, args.k_hi),
        max_instances=args.max_instances,
        device=device,
    )

    module = BatchCopyPaste().to(device)
    if args.compile:
        module = torch.compile(module, fullgraph=True, mode="reduce-overhead")  # pyright: ignore[reportAssignmentType]

    generator = torch.Generator(device="cpu").manual_seed(0)

    def consume(padded: object) -> object:
        return module(padded, generator)

    stmt = batch_stmt(list(batches), consume)

    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    report = run_benchmark(
        label="BatchCopyPaste",
        device=args.device,
        warmup=args.warmup,
        iters=args.iters,
        sub_label=f"b{args.batch_size}_img{args.image_size}_k{args.k_lo}-{args.k_hi}",
        stmt=stmt,
        extra_fields={
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "k_range": [args.k_lo, args.k_hi],
            "max_instances": args.max_instances,
            "compile": bool(args.compile),
            "peak_memory_bytes": (
                int(torch.cuda.max_memory_allocated()) if args.device == "cuda" else 0
            ),
        },
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}  median={report['median_ns'] / 1e6:.2f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
