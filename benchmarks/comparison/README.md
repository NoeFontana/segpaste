# benchmarks/comparison/

Multi-implementation throughput comparison. Pinned in
[`docs/adrs/0016-throughput-comparison.md`](../../docs/adrs/0016-throughput-comparison.md).

**This is a local-dev tool**, not a CI workflow. ADR-0002's
`bench.yml` gate continues to police segpaste self-regression on
every PR.

## Run the smoke grid

```bash
uv sync --group bench
uv run python -m benchmarks.comparison.sweep \
  --device cpu --grid smoke --warmup 5 --iters 20 \
  --out /tmp/smoke.json -v
uv run python -m benchmarks.comparison.aggregate \
  --inputs /tmp/smoke.json --out /tmp/smoke.md
```

A single workload (`B=2, 256², k=1..3`), all three implementations,
~10 seconds on a 2-core machine.

## Run the default 3D sweep

```bash
uv run python -m benchmarks.comparison.sweep \
  --device cpu --grid default --warmup 50 --iters 200 \
  --out throughput.json -v
```

`{B=2, 8, 32} × {256², 512², 1024²} × {k_hi=5, 20}` = 18 workloads ×
~3 implementations. Roughly 25–35 minutes on a 2-core laptop, faster
on a fast desktop CPU.

## Run mmdet (Python 3.12 only)

`mmdet` is **not** a dependency group; mmcv has no Python 3.13 wheel
and uv's universal lock breaks if it's listed. Install manually
inside a 3.12 venv:

```bash
uv sync --group bench --python 3.12
uv pip install --python 3.12 "mmdet>=3.3" "mmengine>=0.10" "mmcv>=2.1"
uv run --python 3.12 python -m benchmarks.comparison.sweep ...
```

If mmdet is unavailable, the sweep emits `status: "skipped"` for
that impl rather than failing.

## GPU sweep

The same harness drives CUDA:

```bash
uv run python -m benchmarks.comparison.sweep \
  --device cuda --grid default --warmup 200 --iters 500 \
  --out throughput-cuda.json -v
```

mmdet automatically reports `status: "skipped"` because its backend is
NumPy/CPU only (`supports_device(cuda) -> False`).

## Acceptance for committed numbers

Per ADR-0016 §7, numbers land on `main` via a normal PR after **three
sequential local runs** with identical arguments satisfy ADR-0002 §ii's
RSD < 3% acceptance:

```
RSD = stddev(m_1, m_2, m_3) / mean(m_1, m_2, m_3) < 0.03
```

where each `m_i` is the `median_ns` of an `ok` cell. Quote the three
runs in the PR description. Commit one of the three JSONs to
`docs/benchmarks/throughput-report-<date>.json` and the rendered
markdown to `docs/benchmarks/throughput-report.md`.

## What the report looks like

`aggregate.py` emits a per-workload ranked table sorted by
`images_per_sec`, with segpaste's speedup factor vs each reference,
and a footer summarising the env block, skipped impls, and the
ADR-0016 caveats (`num_threads=1`, adapter cost excluded, semantics
not equivalent).
