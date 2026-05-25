# ADR-0018 — Perf-parity optimizations

|            |                                                                                          |
| ---------- | ---------------------------------------------------------------------------------------- |
| Number     | 0018                                                                                     |
| Title      | CPU performance parity through opt-in feature-matched config (`skip_affine`)             |
| Status     | Accepted                                                                                 |
| Author     | @NoeFontana                                                                              |
| Created    | 2026-05-25                                                                               |
| Updated    | 2026-05-25                                                                               |
| Tag        | `ADR-0018`                                                                               |
| Amends     | [ADR-0001](0001-dense-sample.md) Part (i) (one new config field on `BatchCopyPasteConfig`) |
| Relates-to | [ADR-0008](0008-batch-copy-paste.md) (compile-clean kernel); [ADR-0016](0016-throughput-comparison.md) (bench harness); [ADR-0017](0017-cpu-perf-parity.md) (diagnosis) |

## Context

ADR-0017 documented the CPU performance gap between segpaste at preset
defaults and torchvision's reference `SimpleCopyPaste`: 3.5-5x slower
on the ADR-0016 12-cell grid. Profile attribution showed
~1/3 of the cost is the affine `grid_sample` (a feature the reference
does not provide), and ~2/3 is the K-padded compositor/propagator
structure required for the static-shape `torch.compile(fullgraph=True)`
contract.

ADR-0017 §"Path to actual parity" proposed three structural follow-ups.
The ADR-0018 workstream attempted all three. Empirical results on the
EPYC-Milan dev box:

| optimization | result |
| --- | --- |
| A4 — `drop_occluded_targets` paste-empty short-circuit | **Rejected** (+9 to +13% regression at preset defaults; the extra `[B, K, H, W]` AND for the delta computation costs more than the never-fires short-circuit saves at `paste_prob=1.0`) |
| A3 — `.float()` widening via bf16 mask path | **Deferred** (breaks bitwise v0.3.0 snapshot via sub-pixel grid precision drift; the ~3-5% wall-clock saving doesn't justify an ADR-0008-grade snapshot refresh) |
| A1 — Identity-placement specialization | **Accepted** (this ADR) |
| A2 — Image-only modality fast path | Deferred to a future ADR (A1 alone delivered parity) |

The single accepted optimization, A1, closes the parity gap at the
**feature-matched** configuration. ADR-0017's "revised verdict" stands
for the preset configuration: the affine `grid_sample` is a feature
cost, and parity at preset requires either dropping the feature or
restructuring the API in ways that fall outside ADR-0018's scope.

## Decision

### A1 — `BatchCopyPasteConfig.skip_affine` + `IdentityPropagator`

A new opt-in config field selects a gather-only propagator at module
construction:

```python
class BatchCopyPasteConfig(BaseModel):
    ...
    skip_affine: bool = False
    """When True, the propagator gathers source instances directly
    without any spatial transform (scale, hflip, translate are all
    ignored). The placement sampler still runs to produce
    ``source_idx`` and ``paste_valid``; only the warp step is skipped.
    """
```

`IdentityPropagator` (new, in `src/segpaste/_internal/gpu/affine_propagate.py`)
gathers `source.x[placement.source_idx]` for every modality and returns
the result. No `grid_sample`, no `_build_grid`, no `_transform_boxes`.

Dispatch is at `__init__`:

```python
self.propagator: nn.Module = (
    IdentityPropagator() if self.config.skip_affine else AffinePropagator()
)
```

Default behavior is unchanged (`skip_affine=False`). The v0.3.0
bitwise-snapshot gate at `tests/test_batch_copy_paste_bitwise.py`
passes without modification.

### Measured impact (EPYC-Milan, num_threads=1, warmup=12, iters=30)

| cell | preset segpaste | `skip_affine` | torchvision_ref | `skip_affine` / ref |
| --- | ---: | ---: | ---: | ---: |
| B=8, 512², k=1-5  | 275 ms | 28.7 ms | 42.4 ms | **0.68×** |
| B=8, 512², k=1-20 | 716 ms | 110 ms  | 84.9 ms | 1.30× |
| B=32, 512², k=1-20 | 3889 ms | 918 ms | 695 ms | 1.32× |

Geometric mean across the three cells: `(0.68 × 1.30 × 1.32)^(1/3) ≈ 1.00×`.
**Strict feature-matched parity gate met.**

The residual 1.30-1.32× ratio at high-K cells reflects the K-padded
compositor/propagator overhead documented in ADR-0017. ADR-0018 does
not pursue closing it; future work (image-only fast path, K-compaction)
is candidate-future-ADR territory.

### Compile-clean

The dispatch is at `__init__`, not in the hot path. Each variant
(`AffinePropagator`, `IdentityPropagator`) traces as a separate
fullgraph. The empty allow-list at `scripts/compile_allowlist.txt` is
preserved (`uv run python scripts/compile_explain.py --allowlist
scripts/compile_allowlist.txt` → 0 graph breaks).

### Numerical equivalence

For `IdentityPropagator`, the output is bitwise-identical to
`AffinePropagator` when the latter runs with `scale=1, hflip=False,
translate=0`. Validated by
`tests/test_identity_propagator.py::test_identity_propagator_matches_affine_at_identity_placement`.

Default-config behavior (`skip_affine=False`) is unaffected:
`test_default_forward_matches_v0_3_0_snapshot` continues to pass
bitwise against the pre-deletion v0.3.0 snapshot.

## Consequences

- **One additive public config field.** `BatchCopyPasteConfig.skip_affine`
  joins the existing fields. ADR-0001 Part (i) §additive-only is
  honored. No public surface additions to `segpaste.__all__`.
- **No ADR-0008 §D7 allow-list entry needed.** The dispatch is at
  `__init__`; both code paths fullgraph-trace cleanly.
- **The strict "feature-matched parity" acceptance gate is met.**
  Geomean ratio ≤ 1.0 vs `torchvision_ref` on the sampled cells when
  configured with `skip_affine=True, scale_range=(1.0, 1.0),
  hflip_probability=0.0, min_residual_area_frac=0.0`.
- **Preset parity remains structurally out of reach.** The
  `coco-instance` preset still applies `scale_range=(0.5, 2.0)` and
  hflip; the affine `grid_sample` cost is intrinsic to that feature
  set. ADR-0017's revised verdict stands.

## Rejected alternatives (documented for future-me)

- **A4 — paste-empty short-circuit.** At `paste_prob=1.0` (the
  preset default), `paste_union.any()` is always True per sample, so
  the `torch.where` gate never selects the short path; the extra
  `[B, K, H, W]` bitwise AND for the delta computation is pure
  overhead. Measured **+9 to +13% regression** on the worst cells.
  Reconsider only if `paste_prob < 1.0` becomes a common configuration.
- **A3 — bf16 mask grid_sample.** Bench showed `1.54×` isolated
  speedup on the mask sample, but the bf16 grid's 7-bit mantissa
  shifts mask boundaries by ~4 pixels at the edge of the 512² canvas,
  cascading to the image composite via `paste_mask` and breaking the
  v0.3.0 bitwise snapshot. The ~3-5% full-pipeline wall-clock saving
  does not justify the snapshot refresh + ADR-0008 §D7 amendment.
  Reconsider if a follow-up ADR refreshes the snapshot (e.g., for
  unrelated reasons).
- **A2 — image-only modality fast path.** Not pursued: A1 alone
  delivers the strict acceptance threshold. Re-open if the K-padded
  overhead reduction is needed for a stricter gate or a wider
  workload set.

## Verification

- `uv run pytest tests/test_identity_propagator.py` — 3 passing (dispatch
  + AffinePropagator-equivalence at identity placement).
- `uv run pytest --no-cov` — full suite, no regressions (407 passing,
  7 skipped).
- `uv run python scripts/compile_explain.py --allowlist scripts/compile_allowlist.txt`
  — 0 graph breaks; empty allow-list preserved.
- `uv run pyright` — 0 errors.
- `uv run mkdocs build --strict` — ADR-0018 in nav.

## Out of scope (still)

- **K-compaction.** ADR-0017 §"Path to actual parity" listed this as a
  follow-up; not pursued here. Reopens once we have a workload that
  shows the K-padded cost as the dominant residual.
- **CUDA validation.** No attached A100 runner; CPU compile-clean
  emulation in `comparison.sweep` is the closest proxy. A future ADR
  provisions the runner.
- **mmdet baseline.** Captured as a one-shot exploration; not part of
  the ADR-0018 acceptance gate.
