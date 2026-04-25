# ADR-0005 — `DenseComposite`: the depth-buffered where-composite primitive

|            |                                                                                 |
| ---------- | ------------------------------------------------------------------------------- |
| Number     | 0005                                                                            |
| Title      | Unified where-composite primitive for all modalities                            |
| Status     | Accepted                                                                        |
| Author     | @NoeFontana                                                                     |
| Created    | 2026-04-22                                                                      |
| Updated    | 2026-04-22                                                                      |
| Tag        | `ADR-0005`                                                                      |
| Relates-to | [ADR-0001](0001-dense-sample.md) Parts (i), (ii), (iv); [ADR-0004](0004-batched-dense-sample.md) |

## Context

After W1 (ADR-0003, ADR-0004) the pipeline is fully `DenseSample`-native on
the instance branch, but the per-modality composite logic in
`augmentation/copy_paste.py` (481 lines) and `processing/placement.py`
(311 lines) was written for instance-only copy-paste with `DetectionTarget`
(now deleted). It hardcodes alpha blending, per-object iteration, a fixed
collision threshold, and post-hoc box recomputation. None of `DenseSample`'s
other active modalities (semantic, panoptic, depth, normals) are usable
without rewriting blend and occlusion logic from scratch each time.

P1 workstreams W2–W4 land the remaining modalities:

- **W2** — instance parity + ClassMix (semantic).
- **W3** — panoptic-CP.
- **W4** — depth-aware paste with metric-depth rescaling.

If each of these workstreams keeps rewriting the same composite arithmetic
on a different modality, ADR-0001 Part (ii)'s invariant matrix will diverge
across branches of a copy-paste tree that *should* share one primitive.

## Decision

Introduce a single **`DenseComposite`** operator (§1) — the §7.2
where-composite — from which every modality-specific augmentation falls
out as a ≤50-line wrapper:

- **`InstancePaste`** (W2) — keeps the existing `CopyPasteAugmentation`
  public signature as a shim.
- **`ClassMix`** (W2) — whole-frame semantic-map mix, no placement.
- **`PanopticPaste`** (W3) — panoptic-id composite with stuff/thing split.
- **`DepthAwarePaste`** (W4) — metric-aware z-buffered composite.

The depth clause degenerates to the Ghiasi alpha composite when `depth` is
`None`, recovering the existing instance-branch behavior exactly
(see §4, parity gate).

### 1. `DenseComposite` primitive

**Location.** `src/segpaste/_internal/composite.py`. Kept under the
`_internal` namespace introduced in ADR-0001 Part (i); **not** added to
`segpaste.__all__`. Promotion is deferred until W3/W4 validate that
`DenseComposite`'s interface generalizes across all five modalities.

**Base class.** `torch.nn.Module`. Modality-presence flags are resolved at
`__init__` (`CompositeConfig.modalities: set[Modality]`) so that
`forward` uses only Python-level constants — `torch.compile` traces cleanly
without branch-on-tensor-value.

**Algorithm.** Given `target, source: DenseSample` and a
`paste_mask: Tensor [H, W] bool`:

1. **Effective mask.**
   - If both `target.depth` and `source.depth` are set:
     `M_eff = paste_mask & ((source.depth < target.depth) | ~target.depth_valid)`.
   - Else: `M_eff = paste_mask`. *(Recovers Ghiasi alpha composite bitwise.)*
2. **Per-modality composite.**
   - Float fields (`image`, `depth`, `normals`):
     `out = M_eff * source + (1 − M_eff) * target`, with `M_eff`
     broadcast-cast to the field dtype.
   - Integer label maps (`semantic_map`, `panoptic_map`):
     `out = where(M_eff, source, target)` — no float arithmetic, no
     interpolation at class boundaries.
   - `instance_masks`: the target's masks are subtracted by
     `M_eff` (occlusion); the source's placed masks are stacked on
     top. Fresh `instance_ids` are allocated `[max_prev+1, …)` so the
     ADR-0001 Part (ii) "fresh instance ids on paste" invariant holds.
   - `depth_valid`: `where(M_eff, source.depth_valid, target.depth_valid)`.
3. **Box recomputation.** Centralized: `boxes_out = masks_to_boxes(inst_masks_out)`.
   Per-mask area `< CompositeConfig.min_composited_area` is dropped —
   ADR-0001 Part (ii) invariant.

**Config.** `CompositeConfig` (frozen pydantic, `extra="forbid"`):
`min_composited_area: int = 50`, `occluded_area_threshold: float = 0.99`,
`modalities: frozenset[Modality]`. Does **not** subsume `CopyPasteConfig`
— the two are composed by `InstancePaste`.

### 2. `PlacementSampler`

**Location.** `src/segpaste/_internal/placement.py`. Replaces the full
contents of `src/segpaste/processing/placement.py` — the three-class
generator/validator/placer split collapses to a single `nn.Module`.

**Interface.** `sample(target_size, source_bbox, existing_boxes, padding_mask)
→ (top, left, paste_mask) | None`. Emits a resolved paste-mask directly in
target coordinates. For W2 the sampler is translation-only (integer
`top, left` offsets); scaled/rotated placement via affine + `grid_sample`
is deferred to W3/W4, where actual scaling is needed and the parity
baseline no longer constrains the arithmetic.

**RNG.** The W2 implementation preserves the current dual-RNG pattern
— `torch.randint` on the non-padding path, `random.randint` on the
padding-aware path — to keep the parity sweep bitwise (see §4).
Unification under a single `torch.Generator` is scheduled for the same
follow-up that introduces `grid_sample`.

### 3. Specializations

**`InstancePaste`** (`src/segpaste/_internal/instance_paste.py`).
Composes `CopyPasteConfig` + `CompositeConfig`; iterates source
objects, calls `PlacementSampler`, delegates blend/occlusion to
`DenseComposite`. The public `CopyPasteAugmentation` class in
`augmentation/copy_paste.py` becomes a thin shim around `InstancePaste`
— signature unchanged, `segpaste.__all__` unchanged.

**`ClassMix`** (`src/segpaste/_internal/classmix.py`). Semantic-modality
specialization. Samples a class-union from the source `semantic_map`
(default-excludes the `255` ignore label), builds a `paste_mask`, and
calls `DenseComposite`. Uses no `PlacementSampler` — `ClassMix` is a
whole-frame paste, which makes it a cleaner "does the primitive
generalize?" test case for a second modality than a second placed paste.

### 4. Parity baseline

The W2 rewrite is gated by a bitwise CPU parity sweep:
`tests/test_dense_composite_parity.py` runs 200 fixed-seed
`(target, sources)` pairs through the current `CopyPasteAugmentation`
output snapshot (`tests/fixtures/composite_baseline.pt`, generated from
commit `23a2d47` immediately before the rewrite lands) vs. the new
`InstancePaste`, and asserts `torch.equal` on image, boxes, labels,
`instance_ids`, and masks.

The fixture is **never** regenerated after W2 lands. The generation
script (`scripts/gen_composite_baseline.py`) is committed alongside
the fixture so the baseline is reproducible if needed, but the
canonical anchor is the serialized file, not the script's re-run output.

CUDA uses a statistical-equivalence gate (area-histogram KS-test)
rather than bitwise parity — `grid_sample` is nondeterministic on
CUDA and `torch.randint` generator streams diverge across devices.
CUDA parity is advisory (skipped when no CUDA is available), not
a merge gate.

### 5. Public surface

W2 makes **no** change to `segpaste.__all__`. All new symbols
(`DenseComposite`, `CompositeConfig`, `PlacementSampler`,
`InstancePaste`, `ClassMix`, `ClassMixConfig`) land under
`segpaste._internal`. Promotion is deferred to a follow-up ADR after
W3/W4 validate the interface. The `CopyPasteAugmentation` shim
preserves the current signature and is the only public entry point
into the composite primitive during 0.9.x.

## Consequences

- `src/segpaste/processing/placement.py` is deleted (contents fully
  replaced by `_internal/placement.py` — not a rename).
- `augmentation/copy_paste.py` collapses from 481 to ~40 lines.
- `tests/test_placement.py` + `tests/test_placement_fuzz.py` are
  deleted, superseded by `tests/test_placement_sampler.py`.
- `tests/test_copy_paste.py` + `tests/test_copy_paste_fuzz.py` keep
  their current shape — they exercise the shim end-to-end and
  provide behavioral regression coverage alongside the bitwise
  parity sweep.
- Benchmarks (`benchmarks/bench_copy_paste.py`) are re-baselined
  post-rewrite; `baseline.json` is updated in the same commit that
  flips the shim. `benchmarks/bench_classmix.py` lands alongside.
- Promoting any `_internal` symbol to the public surface requires an
  ADR-0005 amendment **and** an entry added to
  `tests/test_public_surface.py::_EXPECTED_PUBLIC_API` per ADR-0001
  Part (i).

## Alternatives considered

- **`torchvision.transforms.v2.Transform` base.** The v2 transform
  contract (`make_params` / `transform(inpt, params)`) assumes
  single-input flows; `DenseComposite` is inherently binary
  (`target, source, paste_mask`). Keeping it a plain `nn.Module`
  avoids shoehorning a two-input op into a one-input protocol.
- **Keep per-composite operators separate, share only a config
  base class.** Discarded: ADR-0001 Part (ii)'s invariant matrix
  is exactly the set of constraints a shared primitive enforces
  centrally. Duplicating the depth clause across four
  specializations is the failure mode ADR-0005 exists to prevent.
- **Land `grid_sample` in W2.** Discarded: the parity gate
  forbids arithmetic drift, and W2's placement is translation-only
  in the current implementation. Introducing `grid_sample` for no
  geometric gain would forfeit bitwise parity for nothing.
- **Promote `DenseComposite` to `segpaste.__all__` immediately.**
  Discarded: one-modality validation (instance + semantic) is a
  weak signal that the interface generalizes. The public surface
  is cheap to add to and expensive to remove from (ADR-0001 Part
  (i)); wait for W3/W4 pressure-testing before committing.
