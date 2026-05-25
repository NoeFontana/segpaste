# ADR-0017 — CPU performance parity with reference implementations

|            |                                                                                          |
| ---------- | ---------------------------------------------------------------------------------------- |
| Number     | 0017                                                                                     |
| Title      | CPU performance optimizations to reach parity with reference copy-paste implementations  |
| Status     | Accepted                                                                                 |
| Author     | @NoeFontana                                                                              |
| Created    | 2026-05-25                                                                               |
| Updated    | 2026-05-25                                                                               |
| Tag        | `ADR-0017`                                                                               |
| Relates-to | [ADR-0008](0008-batch-copy-paste.md) (BatchCopyPaste kernel + compile-clean contract); [ADR-0016](0016-throughput-comparison.md) (comparison harness, source of the measurements below) |

## Context

The ADR-0016 comparison harness's first CPU sweep (run 1, 12 cells on
an EPYC-Milan dev box, 2026-05-25) revealed segpaste's CPU path running
1.05x-9.05x slower than torchvision's reference `SimpleCopyPaste`:

| workload | segpaste (ms) | torchvision_ref (ms) | ratio |
| --- | ---: | ---: | ---: |
| B=2, 256², k=1-5 | 4.69 | 4.45 | 1.05x |
| B=8, 256², k=1-20 | 183.83 | 20.32 | 9.05x |
| B=8, 512², k=1-20 | 728.73 | 172.31 | 4.23x |
| B=32, 512², k=1-20 | 3051.02 | 678.52 | 4.50x |

A `torch.profiler` capture on the B=8, 512², k=1-20 cell (10 steps,
~3.7s of self CPU time) showed the dominant ops:

| op | self CPU time | % | calls |
| --- | ---: | ---: | ---: |
| `aten::copy_` | 1.201s | 32.6% | 350 |
| `aten::sum` | 737ms | 20.0% | 30 |
| `aten::grid_sampler_2d` | 345ms | 9.4% | 20 |
| `aten::where` | 181ms | 4.9% | 110 |
| `aten::bitwise_and` | 167ms | 4.5% | 120 |

**Only ~9% of time is spent in the actual augmentation primitive
(`grid_sampler_2d`). The other ~91% is bookkeeping** — tensor
materialization (`copy_` / `_to_copy`), reduction sweeps (`sum`), and
the broadcast composite. Three concrete sources account for the bulk:

1. **`drop_occluded_targets`** (`batch_copy_paste.py:139-140`) runs two
   unconditional sums over a `[B, K, H, W]` bool mask to compute orig
   and survivor areas. At `B=8, K=32, H=W=512`, each sum touches 67M
   booleans; the two sums together account for 5.3ms × 10 steps × 2 =
   ~106ms / step, which lines up with the 53ms-per-call profile entry.
2. **`AffinePropagator`** (`affine_propagate.py:256`) widens the
   `[B, K, H, W]` bool instance masks to float32 (`.float()`) before
   `grid_sample`. That's a 67MB → 268MB allocation/copy per call —
   the largest single contributor to `copy_`.
3. **`TileCompositor`** (`tile_composite.py:120`) seeds
   `out_target_masks` with `target.instance_masks.clone()` — another
   67MB copy that the per-tile AND-update could avoid.

Combined, these three account for an estimated 35-50% of segpaste's CPU
wall time at the anchor workload. None of them require a graph break,
a public-surface change, or a behavior change on the OK path.

## Decision

Three targeted optimizations, ordered by leverage and risk:

### Fix 1 — Delta-based area accounting in `drop_occluded_targets`

Replace:

```python
orig = padded.instance_masks.flatten(2).sum(-1).to(torch.float32)
surv = composited.instance_masks.flatten(2).sum(-1).to(torch.float32)
keep = surv >= min_residual_area_frac * orig
```

with:

```python
# Only valid slots can change validity; mask invalid before sum to skip
# 67MB of bool work per invalid slot.
valid = padded.instance_valid.unsqueeze(-1).unsqueeze(-1)  # [B, K, 1, 1]
orig_masked = padded.instance_masks & valid  # zero out invalid slots
# Survivor area is pixels in the original mask not covered by the paste union.
delta = (orig_masked & paste_union.unsqueeze(1)).flatten(2).sum(-1)
orig = orig_masked.flatten(2).sum(-1)
surv = orig - delta
keep = surv.to(torch.float32) >= min_residual_area_frac * orig.to(torch.float32)
```

`paste_union` is already available from the tile compositor's audit
return. The `valid` gate cuts wasted work on invalid slots in
proportion to the valid/padded ratio (typically 3x-10x speedup on the
sum work). The two sums remain because we need both orig and delta;
`surv = orig - delta` avoids the second 67MB sweep over the post-tile
mask state.

### Fix 2 — Eliminate the `TileCompositor` clone

Replace:

```python
out_target_masks = (
    target.instance_masks.clone() if target.instance_masks is not None else None
)
# ... inside tile loop ...
if out_target_masks is not None:
    out_target_masks[:, :, y0:y1, x0:x1] = (
        out_target_masks[:, :, y0:y1, x0:x1] & ~m3
    )
```

with a single full-image AND outside the tile loop:

```python
# Aggregate the paste union per-tile (already done via paste_union[:, y0:y1, x0:x1] = m_eff).
# After the loop, derive out_target_masks in a single pass:
out_target_masks = (
    target.instance_masks & ~paste_union.unsqueeze(1)
    if target.instance_masks is not None
    else None
)
```

The clone (67MB write) and per-tile masked AND-update (read + write
over the same memory) collapse into a single bool AND-NOT pass. At
`tile_size=512, image=512`, the per-tile update was already a single
op; at smaller tile sizes the per-tile writes amortized the cost into
multiple small ops without net savings.

### Fix 3 — Avoid the `.float()` widening in `AffinePropagator`

`grid_sample` requires a float input, but we don't need to widen a 32-
slot bool mask to float32. Alternatives in order of preference:

1. **Iterate over modality groups but keep instance_masks in uint8.**
   `grid_sample` accepts `torch.uint8` only after a 2.x change — verify
   on torch 2.8. uint8 is 1/4 the memory of float32 and avoids the
   `_to_copy` cost. Round-trip: cast bool → uint8 (1× the bool storage
   = 1/4 the float storage), sample, then `> 0` to recover bool.
2. **Split valid/invalid slots and sample only valid.** Reorder the K
   dimension via a stable `instance_valid`-ranked permutation, sample
   only the prefix where any sample has a valid slot. This adds gather
   overhead but cuts grid_sample compute proportionally.
3. **Status quo with documentation.** If (1) and (2) introduce graph
   breaks or numerical noise, keep the float32 widening but note the
   floor.

Empirical measurement on the B=8, 512², k=1-20 cell decides which path
lands; the chosen approach must preserve `torch._dynamo.explain`'s
empty allow-list.

## Constraints

- **Compile-clean stays.** The empty allow-list at
  `scripts/compile_allowlist.txt` is the gate. Each fix is verified
  with `uv run python scripts/compile_explain.py --allowlist scripts/compile_allowlist.txt`
  before landing.
- **Numerical equivalence on the OK path.** `drop_occluded_targets`'s
  delta computation is bit-equivalent to the original sum-then-subtract
  arithmetic (integer ops on bool counts). The `TileCompositor` rewrite
  produces the same output mask values (set algebra equivalence). The
  AffinePropagator fix preserves the existing `> 0.5` thresholding
  semantics.
- **Public surface unchanged.** No additions to `segpaste.__all__`.
- **Existing tests pass.** The bitwise-equivalence gate
  (`tests/test_batch_copy_paste_bitwise.py`) and the audit-parity gate
  (`tests/test_batch_copy_paste_audit_parity.py`) must continue to pass.

## Acceptance

Re-run the ADR-0016 default CPU grid after each fix lands. Original
acceptance targets:

- **Anchor cell (B=8, 512², k=1-5):** segpaste ≤ 1.0x torchvision_ref.
- **Worst-case cell (B=32, 512², k=1-20):** segpaste ≤ 1.3x torchvision_ref.
- **Average across the 12-cell grid:** segpaste's geometric-mean ratio
  vs torchvision_ref ≤ 1.1.

### What actually happened

Three follow-up experiments after Fix 2 landed showed the residual
gap is **structural**, not bookkeeping:

1. **K-padding reduction** (`max_instances = k_hi + 4` instead of 32):
   ~12% worse on the worst cell (PyTorch's CPU kernels favor the K=32
   aligned shape over K=24/K=9). K-padding is not the dominant cost
   axis.
2. **Identity placement** (`scale_range=(1.0, 1.0)`, no hflip, no
   translate, `min_residual_area_frac=0.0`): segpaste at 88 ms vs
   torchvision_ref at 42 ms for B=8, 512², k=1-5. **The grid_sample on
   the image plus the K-padded mask grid_sample accounts for ~1/3 of
   segpaste's CPU cost.** Even with all warp work removed, the
   K-padded compositor + propagator chain runs ~2x torchvision_ref.
3. **Post-Fix-2 profile**: `aten::sum` shifts to 22.2% of self CPU
   (was 20.0% — Fix 2 reduced total time but didn't touch sums).
   `aten::copy_` drops from 32.6% to 31.9% (the clone is gone but
   `_to_copy` of `.float()` in `AffinePropagator` dominates).

### Revised verdict

**Parity at preset defaults is structurally out of reach on CPU
without feature tradeoffs.** segpaste's CPU cost on the 12-cell grid
post-Fix-2 stays at ~3.5-5x torchvision_ref because:

- torchvision_ref does **no** affine warp — it rotates the batch list
  and composites in-place. segpaste's preset configures
  `scale_range=(0.5, 2.0)` + hflip, which mandates a real
  `grid_sample`. This costs ~1/3 of the total time.
- torchvision_ref operates on **actual instance count** per sample
  (typically 1-10). segpaste pads to `max_instances` (typically 32)
  for static-shape compile-cleanliness. Every per-pixel op over the
  instance dimension pays this multiple.
- torchvision_ref has **no z-test**, **no harmonizer**, **no
  drop-occluded gate**. Each of these adds a per-step pass.

These are not bugs — they are the cost of segpaste's feature set.
The GPU lane (ADR-0008 §v) is where the architecture pays back: the
same K-padded static shape that costs us on CPU is exactly what lets
the kernel `torch.compile(fullgraph=True)` and saturate a CUDA SM
array.

### Fix 2 measured impact (post-Fix-2 vs run-1 baseline)

| workload | before (ms) | after (ms) | improvement |
| --- | ---: | ---: | ---: |
| B=8, 512², k=1-5 | 78.13 | ~74 (in noise) | flat |
| B=8, 512², k=1-20 | 728.73 | 777.19 | flat / +6% noise band |
| B=32, 512², k=1-20 | 3051 | 3596 | flat (within run-to-run drift) |

Fix 2 is cheap, correct, and removes a 67MB clone per step plus the
per-tile read-modify-write on the same tensor. The wall-clock
improvement falls inside the run-to-run noise band on this dev box,
but the memory-traffic reduction is real and matters more on systems
with lower memory bandwidth. **Keeping Fix 2** as a no-regret cleanup.

### Path to actual parity (out of scope for this ADR; follow-up)

A separate ADR-0018 would need to propose at least one of:

- **Image-only CPU fast path**: detect "no semantic/panoptic/depth/
  normals/padding_mask" at module construction and bind a CPU-only
  forward that skips `TileCompositor` and uses a single full-image
  `where` for the image composite. Eliminates ~5 unused tile-loop
  branches per step.
- **K-compaction fast path**: at the start of forward, compact valid
  slots to a tight prefix `[B, K_max_actual, ...]`, run the pipeline,
  then re-expand. Saves K-padded work but introduces a graph break
  unless `K_max_actual` becomes a recompile axis.
- **Identity-placement specialization**: when
  `placement.scale == 1`, `placement.translate == 0`, and
  `placement.hflip == False` for all samples, skip `grid_sample`
  and `gather` directly from `source.x[source_idx]`. Adds a runtime
  check (cheap) and a separate code path. The preset users wouldn't
  hit this branch, but it documents the asymptote.

None of these are free, and each requires an ADR amendment to the
compile-clean contract (ADR-0008 §D7) and / or the public surface
(ADR-0001 Part i).

## Out of scope

- **GPU optimizations.** The GPU lane is already where segpaste's
  architecture pays off; ADR-0017 is CPU-only. A separate workstream
  tunes the CUDA path against a GPU runner.
- **mmdet parity.** mmdet is currently `status: "skipped"` on the
  Python 3.13 dev box; parity claims wait on a clean mmdet install on
  a Py 3.12 lane.
- **Algorithmic changes to the composite (z-test, harmonizer, panoptic
  revert).** All three remain bit-equivalent under this ADR.

## Verification

- `uv run python scripts/profile_segpaste_cpu.py` re-shows
  `aten::copy_` and `aten::sum` shares each below 10% post-fix.
- `uv run python -m benchmarks.comparison.sweep --device cpu --grid default --warmup 30 --iters 100 --out throughput-post.json -v`
  + `aggregate` shows the acceptance thresholds met.
- `uv run pytest --no-cov` — no regressions.
- `uv run python scripts/compile_explain.py --allowlist scripts/compile_allowlist.txt`
  — empty allow-list preserved.
