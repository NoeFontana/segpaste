# ADR-0016 — Throughput comparison across copy-paste implementations

|            |                                                                                          |
| ---------- | ---------------------------------------------------------------------------------------- |
| Number     | 0016                                                                                     |
| Title      | Multi-implementation throughput comparison harness, CPU first, GPU-ready                 |
| Status     | Accepted                                                                                 |
| Author     | @NoeFontana                                                                              |
| Created    | 2026-05-24                                                                               |
| Updated    | 2026-05-24                                                                               |
| Tag        | `ADR-0016`                                                                               |
| Relates-to | [ADR-0002](0002-performance-baseline.md) (CPU baseline, schema v1, regression gate); [ADR-0008](0008-batch-copy-paste.md) §v (GPU lane); [ADR-0015](0015-thirty-minute-integration.md) (migration sources are the comparison set) |

## Context

ADR-0002 pins a self-regression contract for `BatchCopyPaste`: one
workload, one implementation, a 5% median-regression gate against a
committed baseline. That is the right shape for catching internal
regressions, but it cannot answer the question that downstream users
actually ask:

> "How fast is `segpaste.BatchCopyPaste` compared to the implementation
> I would use otherwise?"

The migration guide (ADR-0015) names two such "otherwise" implementations:
- `SimpleCopyPaste` from
  [`pytorch/vision:references/detection/transforms.py:551`](https://github.com/pytorch/vision/blob/main/references/detection/transforms.py)
  — reference training-script code, not part of the installable
  `torchvision.transforms.v2` namespace.
- `mmdet.datasets.transforms.CopyPaste` from
  [`open-mmlab/mmdetection:mmdet/datasets/transforms/transforms.py:2967`](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/datasets/transforms/transforms.py)
  — NumPy-per-sample, the augmentation many production training stacks
  actually run.

A comparison story closes the loop ADR-0015 opens. It also gives the
GPU lane (ADR-0002 §v / ADR-0008 §v) something concrete to point at
once a CUDA runner is available: not just "≥ 2× the segpaste CPU
baseline" but "Nx the reference CPU implementations, measured."

This ADR records the harness design. It deliberately does **not**
amend ADR-0002 — schema v1 and the regression gate stay untouched.
The new artifacts live in a separate `comparison_v1` schema, a
separate result directory, and a manual-dispatch CI lane.

## Decision

### 1. Comparison set

Three implementations enter the matrix:

| Name                | Source                                                                                                  | Install                              | Device support |
| ------------------- | ------------------------------------------------------------------------------------------------------- | ------------------------------------ | -------------- |
| `segpaste`          | `BatchCopyPaste` (this repo)                                                                            | hard dep                             | CPU + CUDA     |
| `torchvision_ref`   | `SimpleCopyPaste` vendored from `pytorch/vision:references/detection/transforms.py` (BSD-3, attributed) | vendored under `benchmarks/_refs/`   | CPU + CUDA     |
| `mmdet`             | `mmdet.datasets.transforms.CopyPaste`                                                                   | optional `[bench-mmdet]` extra       | CPU only (NumPy backend) |

Detectron2 is out of scope: no Python 3.13 wheels, and a CUDA-built
install on the bench runner doubles the workflow cost. A hand-rolled
NumPy floor is also out of scope: every user we'd point at it would
be better served by one of the three above.

### 2. Fairness contract

A `Workload` dataclass (`benchmarks/comparison/workload.py`) generates
the canonical sample data once per `(batch_size, image_size, k_range,
max_instances, seed, device)` tuple. Each `Implementation` provides:

```python
class Implementation(Protocol):
    name: str
    def supports_device(self, device: torch.device) -> bool: ...
    def adapt(self, workload: Workload) -> Sequence[InputBatch]: ...
    def step(self, batch: InputBatch) -> None: ...
```

The adapter converts the shared canonical samples into the
implementation's native input shape:

- `segpaste` → `PaddedBatchedDenseSample`
- `torchvision_ref` → `(list[Tensor[C,H,W]], list[dict[str, Tensor]])`
- `mmdet` → list of `dict(img=np.ndarray, gt_masks=BitmapMasks, gt_bboxes, gt_bboxes_labels)`

The timer (`benchmarks/_harness.py::run_benchmark`) measures `step()`
only — **adapter cost is excluded from the timed window**. Adapter cost
is the per-impl I/O overhead a user already pays in their training
pipeline; the bench measures the augmentation kernel itself.

### 3. Workload sweep

A 3D grid sweep, manual dispatch only:

| Axis        | Values            |
| ----------- | ----------------- |
| Batch size  | `{2, 8, 32}`      |
| Image size  | `{256, 512, 1024}` (square)  |
| `k_hi`      | `{5, 20}` (`k_lo = 1` throughout) |

= 18 workloads × 3 implementations = 54 cells per device. mmdet only
runs on CPU; the CUDA dispatch skips it cleanly via
`supports_device`. The anchor point `(B=8, 512², k=1..5)` is shared
with ADR-0002 so a regression in `segpaste` shows up in both gates.

### 4. Timer reuse

`benchmarks/_harness.py::run_benchmark` is reused verbatim — same
`num_threads=1` pin, same warmup/iters defaults (100/800 CPU,
200/1000 CUDA), same Timer primitive. No changes to its signature.

### 5. Result schema (`comparison_v1`)

A new JSON shape, separate file. ADR-0002's schema v1 is embedded
**per implementation** so downstream tooling can pull a single-impl
report out and feed it to `compare.py` unchanged:

```json
{
  "schema_version": 1,
  "schema": "comparison_v1",
  "workload": {
    "batch_size": 8,
    "image_size": 512,
    "k_range": [1, 5],
    "max_instances": 32,
    "seed": 0,
    "device": "cpu",
    "n_batches": 8
  },
  "env": {
    "python": "3.12.x",
    "torch": "2.8.x",
    "torchvision": "0.23.x",
    "numpy": "2.x.x",
    "mmdet": "3.3.x or null",
    "segpaste_version": "0.2.x",
    "cpu_model": "string or null",
    "runner": "string",
    "commit_sha": "string or null",
    "timestamp": "ISO8601"
  },
  "implementations": [
    {
      "name": "segpaste",
      "status": "ok",
      "report": { /* ADR-0002 schema v1 report */ },
      "images_per_sec": 0.0,
      "batches_per_sec": 0.0
    },
    {
      "name": "mmdet",
      "status": "skipped",
      "skip_reason": "mmdet not installed (install via `uv sync --group bench-mmdet`)",
      "report": null,
      "images_per_sec": null,
      "batches_per_sec": null
    }
  ]
}
```

Derived throughput fields use the canonical wall-clock identity
`images_per_sec = batch_size * 1e9 / median_ns`. They are recorded
alongside `median_ns`, never instead of it.

### 6. Aggregation

`benchmarks/comparison/aggregate.py` reads one or more
`comparison_v1.json` files and emits a markdown report:

- One ranked table per workload (impls sorted by `images_per_sec`).
- A cross-workload pivot showing segpaste's speedup factor vs each
  reference at each `(B, image_size, k_hi)` cell.
- A footer with the env block and a one-line "ok / N skipped"
  summary.

The markdown report is written to
`docs/benchmarks/throughput-report.md` and committed alongside the
JSON. The docs site (mkdocs nav under "Reference") gains a
"Throughput comparison" page that includes it.

### 7. Execution: local-only tool

The throughput comparison is **not** a CI workflow. It runs on a
developer machine via:

```bash
uv sync --group bench
uv run python -m benchmarks.comparison.sweep \
  --device cpu --grid default --warmup 50 --iters 200 \
  --out throughput.json
uv run python -m benchmarks.comparison.aggregate \
  --inputs throughput.json --out throughput-report.md
```

Bench numbers land on `main` via a normal PR after **three sequential
local runs** with identical arguments satisfy ADR-0002 §ii's RSD < 3%
acceptance criterion on each cell's `median_ns`. The PR description
quotes the three runs; one of the JSONs and the rendered markdown are
committed to `docs/benchmarks/throughput-report.md`.

mmdet is installed manually (not a dependency group — see
`pyproject.toml` for the rationale):

```bash
uv pip install "mmdet>=3.3" "mmengine>=0.10" "mmcv>=2.1"  # Python 3.12 only
```

This is **not** a PR gate. ADR-0002's `bench.yml` gate continues to
police segpaste self-regression on every PR. ADR-0016 is the
human-driven, periodic snapshot used to position segpaste against the
reference implementations.

## Consequences

- **No change to ADR-0002.** Existing baseline.json, baseline_gpu.json,
  `bench.yml`, `bench-gpu.yml`, and `compare.py` are untouched. ADR-0002
  schema v1 reports are embedded per-impl in `comparison_v1` so
  toolchains that read v1 work transitively.
- **mmdet is intentionally not a dependency group.** uv's universal
  lock resolves every group across the project's `requires-python`
  range; mmcv 2.x has no Python 3.13 wheel and its source build fails
  on a missing `pkg_resources`. A `[bench-mmdet]` group would break
  the lock on 3.13 dev boxes. Instead, the developer installs mmdet
  manually on Python 3.12 (`uv pip install`); the bench surfaces this
  honestly with `status: "skipped"` reports when the import fails.
- **Vendored code under `benchmarks/_refs/`** is the first vendored
  third-party code in the repo. The vendored file carries the upstream
  BSD-3 license header and a `# Vendored from <commit-sha>` provenance
  comment. Future torchvision-reference updates require a manual
  re-vendor PR with a fresh provenance comment.
- **GPU-readiness baked in**: the same `Workload`, Implementation
  Protocol, sweep, schema, and aggregator serve CPU and CUDA dispatch
  via the `device` field. mmdet's `supports_device(cuda) -> False`
  drops it from the CUDA grid automatically.
- **Adapter cost is excluded from the timed window.** This is the
  correct choice for "augmentation kernel cost" but understates the
  cost users pay if they are not already in the impl's native shape.
  The report's caveats section calls this out explicitly.
- **Implementations are not semantically equivalent.** mmdet's
  config-driven CopyPaste applies different randomness than the
  torchvision reference, which differs from segpaste's preset-driven
  random affine. The bench compares "the wall-clock cost of one batch
  of augmentation under each impl's own semantics." Forcing semantic
  equivalence is research-paper grade and explicitly out of scope.

## Out of scope

- **Bitwise output comparison.** Each implementation has its own
  randomness contract; they will not match. Equivalence is a separate
  workstream.
- **Memory metrics on CPU.** Peak-RSS / allocator stats are noisy on
  the GitHub runner. `peak_memory_bytes` is recorded only on CUDA
  (same convention as ADR-0008 §v).
- **Auto-commit of bench numbers.** Numbers land via a normal PR after
  three local-or-dispatch runs satisfy the RSD acceptance criterion.
  No bot.
- **A regression gate on cross-impl ratios.** segpaste-vs-mmdet speedup
  can drift for reasons unrelated to segpaste (mmdet release, NumPy
  upgrade). Treating the ratio as a gated metric would manufacture
  failures. The ratio is reported; the gate is ADR-0002's
  self-regression only.

## Verification

- `uv run mkdocs build --strict` passes with ADR-0016 in the nav.
- `uv run python -m benchmarks.comparison.sweep --device cpu --grid smoke --out /tmp/run.json -v`
  produces a valid `comparison_v1.json` on the smoke grid (B=2, 256²,
  k=1..3) in under 30 seconds; mmdet entry reports `status: "skipped"`
  with the install hint when mmdet is not present.
- `uv run python -m benchmarks.comparison.aggregate --inputs /tmp/run.json --out /tmp/report.md`
  produces a markdown table with one ranked row per implementation
  plus a caveats section.
- `uv run pytest tests/test_throughput_harness.py` — 12 passing,
  1 skipped (the mmdet smoke test is `importorskip`-gated).
- `uv run pyright` and `uv run ruff check .` pass with the new package
  in their scopes.
