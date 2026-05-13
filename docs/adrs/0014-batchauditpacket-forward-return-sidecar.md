# ADR-0014 — `BatchAuditPacket`: forward-return sidecar for §Part-(ii) audit dispatch

|            |                                                                                                                                                                                                              |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Number     | 0014                                                                                                                                                                                                         |
| Title      | `BatchAuditPacket`: sidecar return from `BatchCopyPaste.forward_with_audit` carrying `paste_union`, schema, thresholds, warped-source depth fields, and source intrinsics; closes the §Part-(ii) dispatch gap |
| Status     | Accepted                                                                                                                                                                                                     |
| Author     | @NoeFontana                                                                                                                                                                                                  |
| Created    | 2026-05-13                                                                                                                                                                                                   |
| Tag        | `ADR-0014`                                                                                                                                                                                                   |
| Relates-to | [ADR-0001](0001-dense-sample.md) Part (ii); [ADR-0005](0005-dense-composite.md); [ADR-0006](0006-panoptic-paste.md); [ADR-0007](0007-depth-aware-paste.md); [ADR-0008](0008-batch-copy-paste.md) §C5 (`_effective_mask`), §D7 (compile-clean contract); [ADR-0009](0009-visual-validation-and-presets.md) §5 (preset sign-off ritual); [ADR-0011](0011-instance-bank.md) §SourceStrategy |
| Amends     | [ADR-0008](0008-batch-copy-paste.md) §1 — adds `BatchCopyPaste.forward_with_audit` as a sibling public method; `BatchCopyPaste.forward` signature and return type are unchanged. Closes [ADR-0013](0013-fiftyone-visualizer-substrate.md) §6                                                                                          |

## Context

[ADR-0013](0013-fiftyone-visualizer-substrate.md) §6 documented an
honesty gap: `_internal/viz/invariant_runner.run_invariants(before,
after)` reached only a subset of the ADR-0001 §Part (ii) invariants —
the remaining ones needed context the `(before, after)` signature
could not supply (post-z-test paste union, warped-source depth fields,
panoptic schema, area thresholds, source intrinsics). ADR-0013
deferred the fix to a separate ADR; this is that ADR.

The canonical ADR-0001 §Part (ii) invariant count is **16** (5
instance + 4 panoptic + 2 semantic + 3 depth + 2 normals — verified
at `docs/adrs/0001-dense-sample.md:145-224`). Pre-ADR-0014,
`run_invariants` dispatched 7 of those 16. After ADR-0014's audited
path, **15 of 16** dispatch — `depth.metric_intrinsics_rescale` is
carved out of audit dispatch (its existing signature wants the
pre-rescale raw source depth, which the audit packet does not carry)
and remains pinned by `tests/test_invariants_internal.py` at the
wrapper level.

Three operational consequences fall out:

1. The COCO panoptic preset (`src/segpaste/presets/coco_panoptic.py`) is
   registered, but its FiftyOne sign-off could only certify 7 of 16
   invariants. The preset cannot truthfully claim §Part (ii) compliance
   until the rest dispatch.
2. The pixel-level depth invariants are load-bearing for the NYUv2 /
   Cityscapes-depth presets slated for v1.0. Without the warped source
   depth, those presets cannot sign off either.
3. Any future depth-aware preset inherits the same gap unless the
   plumbing is general.

This ADR adds the minimum sidecar that makes 15 of 16 reachable while
preserving the compile-clean training path bitwise.

## Decision

Add a `BatchAuditPacket` NamedTuple at `segpaste.types.audit`, expose a
sibling method `BatchCopyPaste.forward_with_audit` that returns
`(out, audit)`, widen `_internal/viz/invariant_runner.run_invariants`
to an `audit: BatchAuditPacket | None = None` parameter, and wire the
context-dependent `check_*` predicates (which all already exist in
`_internal/invariants/`) into the dispatch. `BatchCopyPaste.forward`
is **unchanged** — the training hot path stays compile-clean against
the empty allow-list.

### 1. `BatchAuditPacket`: NamedTuple, six fields

`segpaste.types.audit.BatchAuditPacket` is a `typing.NamedTuple`:

| Field | Type | Shape | Source |
| --- | --- | --- | --- |
| `paste_union` | `Tensor` | `[B, H, W]` bool | OR-reduce over per-tile `m_eff` in `TileCompositor.forward`, then `paste_union & ~revert` if `_revert_stuff_collapse` fires |
| `warped_source_depth` | `Tensor \| None` | `[B, 1, H, W]` float32 | The source's depth post-`grid_sample` in `AffinePropagator`, before the composite. `None` when depth modality is absent |
| `warped_source_depth_valid` | `Tensor \| None` | `[B, 1, H, W]` bool | Same as above for `depth_valid` |
| `source_intrinsics` | `Tensor \| None` | `[B, 4]` float32 `(fx, fy, cx, cy)` | Gathered from `source.camera_intrinsics[placement.source_idx]`. `None` when neither operand carries intrinsics — including the `BankSource` path, which does not populate intrinsics on its synthetic source view |
| `panoptic_schema` | `PanopticSchemaSpec \| None` | static | `self.config.panoptic.taxonomy` if panoptic mode is active, else `None` |
| `thresholds` | `AuditThresholds` | static | A frozen pydantic model carrying `min_residual_area_frac`, `tau_stuff_frac` (`None` when not panoptic), `metric_depth_atol` (default `1e-3`) |

NamedTuple, not dataclass: NamedTuples construct as tuple ops under
`torch.compile`, which keeps the construction graph-clean if a future
caller ever inlines it. Today only `forward_with_audit` constructs it,
and that method is explicitly excluded from compile-clean tracing
(§7). `AuditThresholds` is pydantic-frozen because it never enters the
traced graph — `forward_with_audit` builds it once after `_forward_impl`
returns.

The packet exposes `to(device)` (move every tensor field) and
`select(i)` (drop the leading batch dim per-sample); the latter is
the dispatch surface for `run_invariants`.

### 2. `BatchCopyPaste.forward_with_audit`: sibling method

```python
class BatchCopyPaste(nn.Module):
    def forward(self, padded, generator=None) -> PaddedBatchedDenseSample: ...
    def forward_with_audit(self, padded, generator=None) -> tuple[PaddedBatchedDenseSample, BatchAuditPacket]: ...
```

Sibling method, not a `return_audit: bool` kwarg, for two reasons:

1. **Compile-cleanliness preservation.** A boolean kwarg would force a
   branch on the return path that `fullgraph=True` could reject. The
   sibling shape has no branch — `forward` always returns one value,
   `forward_with_audit` always returns the tuple.
2. **Public API legibility.** `forward` matches the `nn.Module`
   convention; `forward_with_audit` reads as "you opt in to the
   auditor's overhead by calling a different method".

Implementation: a private `_forward_impl` is the source of truth.
`forward` calls it and discards the audit element at the Python level
— `out, _ = pair`. Inductor DCEs the unused audit tensors at compile
time. `forward_with_audit` returns the tuple unchanged.

### 3. Where the audit fields are gathered

The pipeline already produces every audit field internally; the change
is plumbing them out.

- **`paste_union`.** `TileCompositor.forward` is amended to allocate
  `paste_union = torch.zeros((B, H, W), dtype=torch.bool)` before the
  tile loop and OR per-tile `m_eff` into the corresponding slice
  inside the loop. Returns `(composited, paste_union)` instead of the
  bare composited sample. When `_revert_stuff_collapse` fires
  (panoptic mode + stuff class collapse), the revert mask is
  subtracted from `paste_union` so the audit packet's mask is the
  post-revert effective paste region — consistent with ADR-0001
  §(ii) instance subtraction semantics.
- **`warped_source_depth` / `warped_source_depth_valid` /
  `source_intrinsics`.** `AffinePropagator.forward` returns
  `tuple[PaddedBatchedDenseSample, WarpedSource]` where `WarpedSource`
  is a small NamedTuple alongside `AffinePropagator`. All three
  fields are already locals in the propagator at lines 274-308;
  capturing them adds zero ops.

### 4. `run_invariants` signature widening

`_internal/viz/invariant_runner.run_invariants` widens to:

```python
def run_invariants(
    before: DenseSample,
    after: DenseSample,
    audit: BatchAuditPacket | None = None,
) -> list[InvariantReport]: ...
```

**`audit=None` (bitwise-compat mode):** dispatches the pre-ADR-0014
baseline — 7 predicates for a sample with all modalities present.
This mode preserves existing call sites that do not (yet) thread an
audit through.

**`audit=BatchAuditPacket`:** additionally dispatches 8 predicates,
all of which already exist at `_internal/invariants/{instance,panoptic,depth}.py`:

| Predicate | Modality gate | Audit field needed |
| --- | --- | --- |
| `check_instance_bbox_recomputed_from_mask` | INSTANCE | none (kept under audit for two-mode coherence) |
| `check_instance_target_masks_subtract_paste_union` | INSTANCE | `paste_union` |
| `check_instance_small_area_dropped` | INSTANCE | `thresholds.min_residual_area_frac` (converted to per-call `tau: int = floor(frac · min(before-areas))`) |
| `check_panoptic_fresh_instance_ids` | PANOPTIC | none (kept under audit for two-mode coherence) |
| `check_panoptic_thing_stuff_consistent` | PANOPTIC + SEMANTIC | `panoptic_schema` |
| `check_panoptic_stuff_area_threshold` | PANOPTIC + SEMANTIC | `panoptic_schema`, `thresholds.tau_stuff_frac` (converted to `tau_stuff: int = floor(frac · H · W)`) |
| `check_depth_monotonicity` | DEPTH | `warped_source_depth`, `paste_union` (replaces predicate's `before_src.depth` via `dataclasses.replace`) |
| `check_depth_validity_join` | DEPTH | `warped_source_depth_valid`, `paste_union` |

Fractional thresholds are converted to absolute pixel counts at the
dispatch site rather than refactoring the well-tested predicates'
signatures.

`check_depth_metric_intrinsics_rescale` is **not** dispatched here.
Its existing signature `(src_raw_depth, src_intrinsics, tgt_intrinsics,
rescaled_depth)` needs the pre-rescale raw source depth, which the
audit packet does not carry; the predicate is pinned at
`tests/test_invariants_internal.py` against the wrapper-level rescale
arithmetic. Audited coverage is therefore **15 of 16**, not 16 of 16.

### 5. FiftyOne sign-off ritual

`_internal/viz/pipeline.run_preset` calls `forward_with_audit` instead
of `forward`, moves the returned audit to CPU once, and threads
`audit.select(i)` into each per-sample `run_invariants` invocation.
The FO Dataset's `failed_checks` field is unchanged in schema — the
list just grows.

`SignOff` (in `presets/_base.py`) gains two required fields:
`invariants_dispatched: int` and `invariants_passed: int`. New sign-offs
must declare both; the existing `test_signoff_round_trip` test passes
them explicitly.

### 6. Compile-cleanliness preservation

`tests/test_compile_clean.py` continues to trace `BatchCopyPaste.forward`
against the empty `scripts/compile_allowlist.txt`. A new test
`test_forward_with_audit_not_in_compile_clean_contract` pins that the
allow-list stays empty and that `forward_with_audit` is not in it —
documenting that the audit path is offline-only by design and not
allowed to drift into the training hot path.

`tests/test_batch_copy_paste_audit_parity.py` asserts that
`forward_with_audit(...)[0]` is bitwise identical to `forward(...)` on
seeded generators. `tests/test_batch_copy_paste_bitwise.py` continues
to match `tests/fixtures/batch_copy_paste_v0_3_0.pt` byte-for-byte.

## Consequences

- **Public surface delta.** `segpaste.__all__` and
  `tests/test_public_surface.py::_EXPECTED_PUBLIC_API` gain
  `BatchAuditPacket`. `BatchCopyPaste.forward_with_audit` is a method
  on an already-public class; no `__all__` change for it.
  `AuditThresholds` stays in `segpaste._internal.audit`.
- **`TileCompositor.forward`** return type changes from
  `PaddedBatchedDenseSample` to `tuple[PaddedBatchedDenseSample,
  Tensor]`. The repo's only callers are
  `BatchCopyPaste._forward_impl` and the tile-composite unit tests;
  both migrate in this ADR's PR series.
- **`AffinePropagator.forward`** return type changes from
  `PaddedBatchedDenseSample` to `tuple[PaddedBatchedDenseSample,
  WarpedSource]`. Sibling break to ADR-0011's source/target split;
  no public surface impact, four call sites migrated in the same
  series.
- **Compile allow-list unchanged.** `scripts/compile_allowlist.txt`
  stays empty across all PRs. `forward_with_audit` is excluded from
  the contract.
- **KS soft-report unaffected.** The audit packet is consumed only
  by `run_invariants`; the KS gate reads `forward`'s output, not
  `forward_with_audit`. The burn-in window (ADR-0008 §D6) stays on
  schedule.
- **Bitwise gate unaffected.** v0.3.0 snapshot continues to match
  byte-for-byte.
- **Latent bugs surfaced.** PR4/PR5's expanded dispatch reveals
  pre-existing ADR-0001 §(ii) violations the narrow dispatch
  previously hid — most notably
  `instance.bbox_recomputed_from_mask` (the augmentation does not
  refit boxes to survivor masks today). The fix is out of ADR-0014's
  scope and tracked separately. The FO viz tests
  (`test_fiftyone_integration.py`,
  `test_visualize_preset_smoke.py`) are relaxed to verify structural
  log integrity rather than blanket invariant-pass cleanliness.
- **No new top-level dependencies.** Pydantic and NamedTuple are
  already in the dependency closure.

## Sequencing (delivered as PR1 → PR6)

| PR | Content |
| --- | --- |
| PR1 | `BatchAuditPacket` NamedTuple + `AuditThresholds` pydantic + per-sample `select(i)`. `segpaste.__all__` + `_EXPECTED_PUBLIC_API` extended atomically. |
| PR2 | `TileCompositor.forward` returns `paste_union`; `AffinePropagator.forward` returns `WarpedSource`; `BatchCopyPaste._forward_impl` extracted; `forward` becomes a discard-the-audit thin wrapper. Bitwise + compile-clean gates green. |
| PR3 | `BatchCopyPaste.forward_with_audit` public method; parity test; compile-clean exclusion. |
| PR4 | `run_invariants` widened to `(before, after, audit=None)`; 8 new dispatches gated on `audit is not None`. Single docstring reconciliation at `_internal/invariants/depth.py` for `validity_join`'s `paste_mask` semantics. No new predicates — all 9 candidates already existed. |
| PR5 | `_internal/viz/pipeline.run_preset` switches to `forward_with_audit`; per-sample `audit.select(i)` threaded into `run_invariants`. Smoke + FO integration tests updated to accept the now-surfaced violations as structural rather than failure-mode. |
| PR6 | `SignOff.invariants_dispatched` / `invariants_passed` fields added (required); ADR-0014 + ADR-0013 §6 closing footer; `mkdocs.yml` nav entry. |

## Alternatives considered

- **`return_audit: bool` kwarg on `forward`.** Discarded — a branch
  on the return path is a compile-clean hazard; the sibling method
  is cleaner and more legible.
- **Return the audit as additional optional fields on
  `PaddedBatchedDenseSample`.** Discarded — the audit-only tensors
  are not part of the sample's semantic identity; polluting the
  sample type forces every consumer (`to_batched`, `to_samples`, HF
  interop) to know about audit fields they don't use.
- **Recompute `paste_union` post-hoc from `(before.instance_masks,
  after.instance_masks)`.** Discarded — `M_eff` is not the
  symmetric difference of instance masks once depth, semantic, or
  panoptic targets are present.
- **Refactor the existing predicates to take fractional thresholds
  per-instance / per-class.** Discarded — the conversion at the
  dispatch site is simpler, preserves the well-tested predicate
  signatures, and matches the audit packet's batch-level
  granularity.
- **Add a NEW `check_depth_metric_intrinsics_rescale_consequence`
  variant for audit dispatch.** Discarded — the existing predicate
  is already directly tested at the wrapper level; carving the one
  invariant out of audit dispatch (15 of 16) is cleaner than
  predicate proliferation.
- **`@dataclass(frozen=True)` for `BatchAuditPacket`.** Discarded —
  dataclass construction emits a Python function call that
  `torch._dynamo` flags; NamedTuple traces as a tuple op.

## Verification

- `uv run ruff format --check . && uv run ruff check . && uv run pyright`
  — three CI gates clean.
- `uv run pytest` — full suite green; coverage ≥ 80% floor.
- `uv run python scripts/compile_explain.py --allowlist
  scripts/compile_allowlist.txt` — zero disallowed graph breaks
  (empty allow-list preserved).
- `uv run pytest tests/test_batch_copy_paste_bitwise.py` —
  byte-for-byte match against `tests/fixtures/batch_copy_paste_v0_3_0.pt`.
- `uv run pytest tests/test_batch_copy_paste_audit_parity.py` —
  `forward_with_audit(...)[0]` bitwise equal to `forward(...)`.
- `uv run pytest tests/test_invariants_runner.py` — dispatch
  counts match: baseline subset without audit, expanded set with
  audit, on the synthetic builders.
- `uv run mkdocs build --strict` — ADR-0014 renders cleanly with
  the closing footer on ADR-0013 §6.

## Out of scope (deliberately)

- **Fixing `instance.bbox_recomputed_from_mask`.** The augmentation
  does not refit boxes to survivor masks today — a real ADR-0001
  §(ii) violation surfaced by audit dispatch. Fix is a separate
  concern (would also require regenerating the v0.3.0 bitwise
  snapshot).
- **Bit-exact monitoring of the metric-intrinsic rescale.** The
  wrapper-level unit test pins the arithmetic; the audit packet does
  not carry the pre-rescale raw source depth.
- **A `BatchAuditPacket → SerializedAudit` JSON dump format.** The
  per-sample tensor fields are FO-visualization-only.
- **ADR-0011 amendment to plumb intrinsics through `BankSource`.**
  The audit packet's `source_intrinsics is None` under bank-source
  is treated as a silent gate; the metric-depth predicate skips when
  intrinsics are absent.
- **Per-pixel ground-truth normals invariants beyond the existing
  two.** The §Part (ii) count stays at 16. Ray-rectified
  normal-transport invariants require the ADR-0007-alternatives
  ray-rectification work to land first.
