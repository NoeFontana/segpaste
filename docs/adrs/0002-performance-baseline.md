# ADR-0002 — Performance Baseline and Benchmark Harness

|            |                                                      |
| ---------- | ---------------------------------------------------- |
| Number     | 0002                                                 |
| Title      | Performance baseline and benchmark harness for `CopyPasteCollator` |
| Status     | Accepted                                             |
| Author     | @NoeFontana                                          |
| Created    | 2026-04-20                                           |
| Updated    | 2026-04-20                                           |
| Tag        | `ADR-0002`                                           |

Every Phase 1 (P1) change that touches `CopyPasteCollator`, the copy-paste
augmentation pipeline, or the placement logic should reference `ADR-0002`
when it ships a performance claim. Superseding a decision means writing a
new ADR that supersedes this one, not quietly drifting.

## Context

Phase 1's exit criterion requires a ≥2× throughput improvement over the
v0.9.x `CopyPasteCollator` baseline. ADR-0001 explicitly scoped the
performance budget as out of scope — this ADR closes that gap so the 2×
target is a measurement, not an argument.

The repository prior to this ADR has no benchmark harness, no committed
baseline, and no regression gate. Any perf claim in a P1 PR would be
unverifiable without this infrastructure.

This ADR pins interface-level decisions only (workload, measurement
contract, acceptance and gating policy, refresh policy). The GPU lane,
durable time-series storage, and per-operation microbenchmarks are
deliberately deferred.

---

## Part (i) — Measurement contract

### Canonical workload

The `CopyPasteCollator` baseline is measured against a single synthetic
workload defined by `benchmarks/_fixture.py::build_batch`:

- **Batch size:** `8`.
- **Image size:** `512 × 512` (see note below).
- **`k` pasted instances per image:** $k \sim \mathcal{U}\{1, \dots, 5\}$
  (inclusive endpoints, 5 possible values).
- **Image dtype:** `uint8`, random values in `[0, 256)`.
- **Bounding boxes:** xyxy `float32`, width and height each drawn from
  $\mathcal{U}[64, 256]$, position clamped so boxes stay inside the image.
- **Labels:** `int64`, drawn from $\mathcal{U}[1, 80]$ (COCO-class cardinality).
- **Masks:** `[k, H, W]` `bool`, the interior of each bounding box (guarantees
  `area >= 1` for the canonical config).

Reproducibility: each sample is seeded from `seed * 8191 + i`, and the
harness pre-builds `64` distinct batches and cycles through them by index
so the 2000 measurement iterations are bit-identical across runs with the
same seed.

### Image size deviation from the original brief

The initial P0.C brief called for `1024²` with 500 warmup + 2000
measured iterations. On `ubuntu-latest` (GitHub's 2-core default
runner) a single collator call at `1024²` measures ~500 ms locally,
which projects to ~22 minutes for a full pass at the original counts
— too close to the 25-minute job timeout and nearly 40 minutes on a
colder 2-core runner. Two deviations from the brief resolve this:

1. **`512²` replaces `1024²`** as the canonical image size, bringing
   per-iter cost to ~100 ms. `512²` is also a realistic training
   resolution (Mask R-CNN baseline, LVIS short side).
2. **Warmup is 100 iterations; measurement is 800 iterations** (down
   from 500 / 2000). 800 samples is ample for a stable median and
   IQR on wall-clock data, and the 100-iter warmup is sufficient on
   CPU where there is no cuDNN autotune to settle.

Combined, a full pass is ~1.5 minutes instead of ~22. The Part (ii)
acceptance rule (three runs with `IQR/median < 3%`) is applied under
these reduced counts.

`512²` is also a realistic training resolution (Mask R-CNN baseline,
LVIS short side). A future runner upgrade or a GPU lane (see Part iv)
may motivate a new ADR that moves the canonical size back to `1024²`.

### Timer usage

`torch.utils.benchmark.Timer` is the sole measurement primitive:

- `num_threads=1` pins the PyTorch thread pool for the measurement
  window so runner-to-runner core-count differences do not leak into
  the metric.
- **Warmup:** 100 calls to `stmt()`, discarded. On CUDA, a
  `torch.cuda.synchronize()` is emitted once after warmup.
- **Measurement:** 800 calls via `timer.timeit(1).median`, collected
  as raw per-iter nanoseconds.
- `blocked_autorange` is deliberately **not** used: it batches inner
  iterations and collapses them into a single `Measurement`, which
  prevents IQR computation.

### JSON schema (version 1)

Every bench run emits a JSON report with the following shape. The
`schema_version` field is bumped when any field is removed or changes
type; additive changes do not bump it.

```json
{
  "schema_version": 1,
  "label": "CopyPasteCollator",
  "device": "cpu",
  "cpu_model": "string or null",
  "batch_size": 8,
  "image_size": 512,
  "k_range": [1, 5],
  "warmup": 100,
  "iters": 800,
  "median_ns": 0,
  "q1_ns": 0,
  "q3_ns": 0,
  "iqr_ns": 0,
  "iqr_over_median": 0.0,
  "python": "3.12.x",
  "torch": "2.8.x",
  "torchvision": "0.23.x",
  "segpaste_version": "0.9.x",
  "commit_sha": "string or null",
  "runner": "string",
  "timestamp": "ISO8601"
}
```

---

## Part (ii) — Acceptance and gating

### Runner pinning

The committed `baseline.json` and the `bench-cpu-pr` gate both run on
`ubuntu-latest`. A baseline captured on any other runner (dev box,
self-hosted, `ubuntu-latest-large`) is not comparable and must not be
committed. When GitHub changes the default `ubuntu-latest` image in a
way that shifts the median, the baseline is refreshed per Part (iii).

### Baseline acceptance

A new `benchmarks/baseline.json` is accepted when **three** sequential
local runs with identical arguments satisfy

$$\mathrm{RSD}_{\text{median}} = \frac{\mathrm{stddev}(m_1, m_2, m_3)}{\mathrm{mean}(m_1, m_2, m_3)} < 0.03$$

where $m_i$ is the `median_ns` of the $i$-th run. One of the three is
committed verbatim; the other two are retained as evidence in the PR
description.

**Why run-to-run RSD of medians and not within-run IQR/median.** The
collator has substantial data-dependent per-call variance
(`torch.equal` early termination, variable object counts after
placement, mask-subtraction cost scaling with pasted area). Within a
single run, `iqr_over_median` is typically 15–25% — that is a property
of the workload, not of the measurement. The median itself, however,
is a stable run-to-run statistic at this iteration count. The
regression gate in Part (ii) compares medians, so the acceptance rule
measures the right thing: run-to-run stability of the quantity the
gate uses. `iqr_over_median` is retained in the JSON as a diagnostic
for noticing structural workload shifts that change distribution
shape without moving the median.

### Regression gate

A PR fails the `bench-cpu-pr` job when:

$$\frac{\text{median\_ns}_\text{current} - \text{median\_ns}_\text{baseline}}{\text{median\_ns}_\text{baseline}} > 0.05$$

i.e. a median regression strictly greater than 5%. The baseline is the
file on the PR's base branch (`main`), not the head. `batch_size`,
`image_size`, and `device` must match between baseline and current —
mismatches exit with code 2 and must be resolved by refreshing the
baseline.

### Label escape hatches

Two PR labels alter the gate's behavior:

- **`skip-perf-gate`** — the comparison step is skipped entirely; the
  harness still runs and the artifact is still uploaded. Reserved for
  (a) the P0.C PR itself (which creates the baseline), (b) PRs that
  refresh the baseline (see Part iii), and (c) PRs that intentionally
  land perf-affecting refactors whose baseline will be refreshed in
  a follow-up. Use requires a justification in the PR description.
- **`retry-on-perf-flake`** — on `github.run_attempt > 1` (i.e. a
  re-run), the comparison step is skipped. First-attempt runs still
  enforce the gate. This is a one-shot escape hatch for genuine runner
  noise; repeated use indicates the gate is too tight and should be
  widened via a new ADR (additive change).

---

## Part (iii) — Baseline refresh policy

The baseline is refreshed only via an opt-in PR that:

1. Is labeled `skip-perf-gate`.
2. References `ADR-0002` in the commit message or PR description.
3. Includes, in the PR description, the three local runs' `median_ns`
   and `iqr_over_median` values from Part (ii)'s acceptance criterion.
4. Touches `benchmarks/baseline.json` and only that file in its commit
   (i.e. refresh is not bundled with feature work).

A refresh that changes `batch_size`, `image_size`, `k_range`, `warmup`,
or `iters` is a **breaking change** to the contract and requires a new
ADR that supersedes this one. Refreshes that only reflect runner-image
updates, torch-version bumps, or unchanged-workload re-captures are
additive and land under this ADR.

---

## Part (iv) — Out of scope

- **GPU / CUDA lane.** The harness's `--device cuda` code path is
  implemented and testable locally. The CI lane is deferred to P0.D,
  which provides the A100 runner. Adding the `bench-gpu` job and its
  gate is an additive change under this ADR.
- **Durable time-series storage.** Nightly reports land as GitHub
  Actions artifacts with 14-day retention. A follow-up ADR pins the
  long-term storage mechanism (orphan branch, external TSDB, etc.)
  once Phase 1 demonstrates the metric is load-bearing.
- **Per-operation microbenchmarks.** The baseline is end-to-end on
  `CopyPasteCollator.__call__`. Probing `_try_place_single_object`,
  `_blend_object_on_target`, etc. is an internal-profile activity and
  does not belong in the regression gate.
- **Memory / allocator metrics.** Only wall-clock median and IQR are
  pinned. Peak-RSS and allocator-stats are useful diagnostics but
  would broaden the contract beyond what P1's 2× throughput criterion
  needs.

---

## Verification

- `uv run mkdocs build --strict` passes with this ADR in the nav.
- `uv run python -m benchmarks.bench_copy_paste --device cpu --out /tmp/r.json`
  runs end-to-end locally.
- `uv run python -m benchmarks.compare --baseline benchmarks/baseline.json --current /tmp/r.json`
  exits 0 on an unchanged workload.
- `uv run pyright` and `uv run ruff check .` pass with `benchmarks/`
  included in their scopes.

## Status and supersession

- **Accepted** when this file lands on `main` with `Status: Accepted`
  in the header and `benchmarks/baseline.json` is committed alongside
  it.
- Any later decision that contradicts Part (i)–(iii) requires a new
  ADR (`ADR-000N`) that explicitly supersedes this one, updating this
  file's `Status` to `Superseded by ADR-000N`.
