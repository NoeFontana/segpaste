# ADR-0004 — `instance_ids`, frozen `DenseSample`, and `BatchedDenseSample`

|            |                                                                 |
| ---------- | --------------------------------------------------------------- |
| Number     | 0004                                                            |
| Title      | Instance identity, `DenseSample` immutability, batched container |
| Status     | Accepted                                                        |
| Author     | @NoeFontana                                                     |
| Created    | 2026-04-22                                                      |
| Updated    | 2026-04-22                                                      |
| Tag        | `ADR-0004`                                                      |
| Supersedes | [ADR-0001](0001-dense-sample.md) Part (iii) — type-table and `DenseSample` mutability only |

## Context

P1 workstream W1 migrates the augmentation pipeline off `DetectionTarget` and
onto `DenseSample` end-to-end (see [ADR-0003](0003-hard-deprecation-stance.md)
for the deletion schedule). Three gaps in ADR-0001 Part (iii) surfaced during
W1 design and need pinning before the migration commit lands:

1. **Per-instance identity.** ADR-0001's `DenseSample` schema carries
   `labels: torch.Tensor [N] int64` (class ids) but no per-instance id. The
   panoptic composite (later P1 workstream) needs stable, unique instance ids
   to satisfy the "fresh instance ids on paste" invariant in ADR-0001 §(ii).
   Synthesizing ids at composite time would break idempotence and make the
   paste-union invariant harder to verify. Pinning identity as a first-class
   field closes the gap before more composites land.
2. **Mutability.** ADR-0001 Part (iii) declared `DenseSample` as
   `@dataclass(slots=True)`. The pipeline is functional in practice
   (`CopyPasteAugmentation.transform` returns a new container; no in-place
   mutation), so freezing the class has no code cost and gives
   `torch.compile` a cleaner aliasing story (W5 relies on this).
3. **Batched output.** `CopyPasteCollator.__call__` today returns a loosely
   typed `dict[str, Tensor | list[Tensor]]`. After W1, the internal pipeline
   is `DenseSample`-native; the collator output should be a typed container
   with the same shape as the per-sample type. An untyped dict prevents
   pyright from catching field-rename regressions across the surface.

## Decision

### Per-instance identity

Add `instance_ids: torch.Tensor | None` to `DenseSample`, shape `[N]`, dtype
`torch.int32`. Required when `instance_masks is not None` (co-optional with
`instance_masks`); `None` otherwise. Ids are unique within a sample but not
across a batch. The `from_detection_target` legacy bridge (removed at the end
of W1) synthesizes `torch.arange(N, dtype=torch.int32)` because
`DetectionTarget` carries no identity information.

Update the ADR-0001 Part (iii) field table to include `instance_ids` between
`labels` and `instance_masks`. Pasted instances receive fresh ids in the
range `[max_prev + 1, max_prev + 1 + k)` where `max_prev` is the largest
surviving-target id (or `-1` when the target was empty). This matches the
"fresh ids" wording already reserved in ADR-0001 §(ii) for the panoptic
composite; extending it to the instance composite keeps the two pathways
aligned.

### `DenseSample` is frozen

`DenseSample` is `@dataclass(frozen=True, slots=True)`. All modifications
return a new instance. `__post_init__` validation continues to run under
`@skip_if_compiling`. No call site in `src/segpaste/` mutates a
`DenseSample` field after construction (verified during W1 exploration);
the flip is source-compatible.

### `BatchedDenseSample`

A new frozen dataclass, pinned into the public surface alongside
`DenseSample`, is the canonical return type of `CopyPasteCollator.__call__`.

| Field | Type | Shape | Stacked/ragged |
| --- | --- | --- | --- |
| `images` | `tv_tensors.Image` | `[B, C, H, W]` | Stacked — LSJ enforces per-sample `(C, H, W)` homogeneity |
| `boxes` | `list[tv_tensors.BoundingBoxes]` | `B × [N_i, 4]` | Ragged |
| `labels` | `list[torch.Tensor]` | `B × [N_i]` int64 | Ragged |
| `instance_masks` | `list[InstanceMask] \| None` | `B × [N_i, H, W]` bool | Ragged; co-optional with `instance_ids` |
| `instance_ids` | `list[torch.Tensor] \| None` | `B × [N_i]` int32 | Ragged |
| `semantic_maps` | `SemanticMap \| None` | `[B, H, W]` int64 | Stacked |
| `panoptic_maps` | `PanopticMap \| None` | `[B, H, W]` int64 | Stacked |
| `depth` | `torch.Tensor \| None` | `[B, 1, H, W]` float32 | Stacked |
| `depth_valid` | `torch.Tensor \| None` | `[B, 1, H, W]` bool | Stacked |
| `normals` | `torch.Tensor \| None` | `[B, 3, H, W]` float32 | Stacked |
| `padding_mask` | `PaddingMask \| None` | `[B, 1, H, W]` bool | Stacked |
| `camera_intrinsics` | `list[CameraIntrinsics] \| None` | `B` elements | Python-object list |

`BatchedDenseSample.from_samples(list[DenseSample]) -> BatchedDenseSample` is
the canonical constructor; it raises on inconsistent `(H, W)` across
samples or modality-active mismatch (e.g. sample 0 sets `semantic_map` and
sample 1 does not). `to_samples()` is the inverse. `__post_init__` validation
runs under `@skip_if_compiling`.

An empty batch (`B == 0`) is valid: stacked fields are zero-sized tensors
with the correct rank; ragged-list fields are empty lists.

### Public surface delta

`segpaste.__all__` gains exactly one symbol: `BatchedDenseSample`. The
pinned tuple in `tests/test_public_surface.py::_EXPECTED_PUBLIC_API` must
grow by the same symbol atomically with the `__all__` edit. All other
existing public names are preserved.

## Consequences

- **`CopyPasteAugmentation.transform` signature changes** to
  `(DenseSample, list[DenseSample]) -> DenseSample`. `CopyPasteCollator.__call__`
  changes to `(list[DenseSample]) -> BatchedDenseSample`. `CocoDetectionV2.__getitem__`
  changes to return `DenseSample`. Consumers pin `segpaste<0.9` to retain the
  old shapes — same migration story as ADR-0003.
- **Benchmark baseline in `benchmarks/baseline.json` becomes
  apples-to-oranges** once the collator input type flips. W1's PR rides the
  `skip-perf-gate` label; a follow-up refresh PR captures a new baseline
  under ADR-0002 §Part (iii) rules.
- **ADR-0001 Part (iii) field table is amended by reference.** ADR-0001's
  header Status line cites this ADR alongside ADR-0003. Part (ii) invariants,
  Part (iv) seeding, and every other Part (iii) clause (`Modality`,
  `PanopticSchema`, `CameraIntrinsics`, tv-dispatch shim rationale) remain in
  force unchanged.

## Status and supersession

- **Accepted** when this file lands on `main` with `Status: Accepted`.
- ADR-0001's header is updated in the same commit to cite this ADR in the
  `Status` line (Part iii type-table and `DenseSample` mutability only).
- Any later decision that removes `instance_ids`, un-freezes `DenseSample`,
  or changes `BatchedDenseSample`'s pinned layout requires a new ADR that
  explicitly supersedes this one.

## Verification

- `uv run mkdocs build --strict` passes with this ADR in the nav.
- ADR-0001 renders with the updated `Status` cross-linking this ADR.
