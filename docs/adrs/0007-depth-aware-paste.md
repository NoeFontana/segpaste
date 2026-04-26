# ADR-0007 — `DepthAwarePaste`: depth z-buffer paste with intrinsics handling

|            |                                                                                         |
| ---------- | --------------------------------------------------------------------------------------- |
| Number     | 0007                                                                                    |
| Title      | Depth-modality composite with intrinsics rescale and h-flip normals transport           |
| Status     | Accepted                                                                                |
| Author     | @NoeFontana                                                                             |
| Created    | 2026-04-23                                                                              |
| Updated    | 2026-04-23                                                                              |
| Tag        | `ADR-0007`                                                                              |
| Relates-to | [ADR-0001](0001-dense-sample.md) Parts (ii), (iii); [ADR-0005](0005-dense-composite.md); [ADR-0006](0006-panoptic-paste.md) |

## Context

ADR-0001 Part (ii) pins five depth/normals invariants — monotonicity on
effective paste pixels, validity join, metric intrinsics rescale;
unit-norm-on-valid for normals, and right-down-forward camera-frame
convention. The predicates are implemented in `tests/invariants/depth.py`
and `tests/invariants/normals.py`, with five
`InvariantRow(Modality.DEPTH|NORMALS, …, xfail=True)` entries in
`tests/test_invariant_matrix.py:201-205` waiting for a composite that
satisfies them.

ADR-0005 landed `DenseComposite.forward` with image, instance, and
semantic branches; ADR-0006 added the panoptic branch in W3. Depth and
normals remain untouched: `forward` never emits `depth`, `depth_valid`,
or `normals`, though `_effective_mask` already performs the z-test
(`composite.py:102-113`). No depth-specialized wrapper exists.

W4 closes that gap. Unlike W2 (anchored to a pre-rewrite bitwise parity
snapshot) but like W3, there is no predecessor implementation — W4 is
defined from first principles against the §(ii) invariants, two
hand-constructed analytical golden fixtures (monotonicity + intrinsics
rescale), and a forward-gate 200-seed snapshot.

## Decision

Add `_composite_depth` and `_composite_normals` branches to
`DenseComposite` and land `DepthAwarePaste` as its specialization under
`src/segpaste/_internal/depth_paste.py`. `DepthAwarePaste` reuses the W2
`PlacementSampler` for translation-only placement, preprocesses the
source with an optional h-flip (with a normals x-sign-flip), performs
the metric intrinsic rescale when active, and delegates the per-pixel
write to `DenseComposite`. The public surface is unchanged —
`segpaste.__all__` stays frozen, `DepthAwarePaste` stays under
`_internal` per ADR-0005 §5.

### 1. `metric_depth` lives on `DenseSample`, not on the wrapper config

Per-sample state pairs naturally with the existing `camera_intrinsics`
field on `DenseSample`. A dataloader can mix metric and affine samples
without constructing two composites. `DenseSample.__post_init__`
enforces: if `metric_depth=True` and `depth` is set, `camera_intrinsics`
must also be set — silent fallback to identity intrinsics is explicitly
forbidden.

`DepthAwarePaste.transform` additionally enforces at runtime that both
operands share the same `metric_depth` flag; mismatched flags raise
`ValueError`. Composing a metric source into an affine target (or vice
versa) would mix two incommensurable depth scales.

ADR-0001 Part (iii) is amended to add `metric_depth: bool = False` to
the `DenseSample` field table and to document the cross-field invariant
in the rationale.

### 2. Validity semantics: target-dominant outside `M_eff`, conjunctive inside

The intended validity formula is piecewise: outside `M_eff`,
`V_out = V_tgt` (un-touched target pixels retain their validity);
inside `M_eff`, `V_out = V_src ∧ V_tgt` (both must be valid for the
pasted pixel to be trusted). The previous ADR-0001 §(ii) wording
("pixelwise AND everywhere") would have let a source's invalid region
invalidate an un-touched target pixel — an unintended regression. ADR-0001
§(ii) is amended to state the piecewise formula, and
`tests/invariants/depth.py::assert_depth_validity_join` is updated to
take `effective_paste_mask` and assert the piecewise formula.

#### 2a. Generalization to all per-pixel validity signals (ADR-0008 amendment)

The piecewise validity formula is not specific to depth. It applies to
every per-pixel validity signal carried on a `DenseSample`. Two
instances exist today:

* `depth_valid` — gates pixel-level depth measurements (this ADR's §2).
* `image_valid := ~padding_mask` — derived from `PaddingMask`. Marks
  which pixels of a sample carry real image content rather than
  pad introduced by `FixedSizeCrop` / LSJ. Composites must not pull
  source-pad zeros into pasted regions, and placements must not be
  drawn over target pad. Both gates fold into `M_eff` symmetrically:
  `M_eff = paste_mask ∧ (z-test) ∧ image_valid_src` (ADR-0008 §C5,
  `_internal/composite.py::_effective_mask`,
  `_internal/gpu/tile_composite.py::_effective_mask`). The placement
  side consults the equivalent `valid_extent` reduction
  (`BatchCopyPaste._valid_extent`) so translates land inside the
  target's valid rect, and discards source rows whose bbox extends
  past the source's valid extent.

Future per-pixel validity signals (e.g. semantic-confidence mask,
amodal-occlusion mask) plug into the same machinery: warp under the
same `grid_sample(mode="nearest", padding_mode="zeros")` template used
for `depth_valid`, AND into `M_eff` at composite time, optionally
contribute to `valid_extent` at placement time. No new ADR is required
for additional per-pixel signals — this generalization is the
architectural commitment.

### 3. Depth composite reuses `_effective_mask`

The z-test is already correct in `_effective_mask` (`composite.py:102-113`):
inside the placement paste mask, source wins iff
`d_src < d_tgt ∨ ~V_tgt`. Monotonicity
(`d_out = min(d_src, d_tgt)` inside `M_eff`) reduces to
`torch.where(m_eff.unsqueeze(0), d_src, d_tgt)` — no explicit
`torch.minimum` required, because `m_eff` already encodes the z-test.
This mirrors `_composite_semantic` and `_composite_panoptic` exactly.

`_composite_normals` is the same shape:
`torch.where(m_eff.unsqueeze(0), n_src, n_tgt)`. Inputs are unit-norm by
precondition; per-pixel selection preserves unit-norm without any
renormalization. No interpolation is introduced by the composite at
pixel granularity.

### 4. Metric intrinsic rescale: geometric mean of focal lengths

When `metric_depth=True` on both operands, the wrapper rescales the
source depth before constructing the synthetic source fed into
`DenseComposite`:

```
d_src_rescaled = d_src * sqrt(fx_t * fy_t) / sqrt(fx_s * fy_s)
```

For isotropic pixels (`fx == fy`), this reduces to the brief's
`f_t / f_s` ratio. For non-square pixels, the geometric mean handles
both axes symmetrically. This is the Metric3D-v2 canonical-camera trick
productionized per the source report §3c.

The rescale runs once in `DepthAwarePaste.transform`; the composite
itself is intrinsics-agnostic.
`tests/invariants/depth.py::assert_depth_metric_intrinsics_rescale` is
updated from its placeholder `fx`-only ratio
(`depth.py:66-67`) to match.

### 5. Validity-join reconciliation with ADR-0001

Because the previous ADR-0001 wording was wrong (§2 above),
`assert_depth_validity_join` was written against a false predicate. W4
changes both the ADR-0001 §(ii) text and the invariant body in the same
commit so the `xfail=False` flip is consistent. Every other caller of
`assert_depth_validity_join` is the fuzz / matrix harness — no silent
pass-through.

### 6. Blend-mode restriction enforced at the type level

`DepthAwarePasteConfig.blend_mode: Literal["alpha"] = "alpha"`.
Constructing `DepthAwarePasteConfig(blend_mode="gaussian")` raises
`pydantic.ValidationError` — no custom `ConfigurationError` class is
needed. Matches the existing `CopyPasteConfig.blend_mode: Literal["alpha"]`
pattern (`src/segpaste/config.py:23-24`).

The source-report §3d reasoning is specific: pasted depth discontinuities
align with real scene depth edges and match gradient-matching losses,
whereas Gaussian feathering would create synthetic depth ramps that are
neither plausible nor in-distribution for monocular-depth models.

### 7. H-flip as wrapper-level preprocessing

`DepthAwarePaste._maybe_hflip_source(source, rng)` flips the source
atomically when `config.hflip_probability` fires:
image, instance_masks, depth, depth_valid, normals, semantic_map,
panoptic_map, and boxes all get the standard torchvision h-flip; the
normals x-component is sign-flipped (`normals[0] = -normals[0]`). This
is the single correct camera-frame transformation under h-flip in the
right-down-forward convention.

Rotation and translation are explicitly out of scope for P1 (source
report §8 defers ray-rectified normal transport). The wrapper does
not accept a rotation parameter; the only geometric transform on
normals is the h-flip sign-flip. `PlacementSampler` is untouched,
preserving W3's parity gate.

### 8. Debug validity-join assert

`DepthAwarePasteConfig.debug_assert_validity: bool = False`. When set,
post-composite asserts `V_out` matches the piecewise formula. Default
off; flagged on by the fuzz test harness. Mirrors
`PanopticPasteConfig.debug_assert_bijection`.

### 9. NYUv2 AbsRel regression is out of scope

A reference SegFormer-like fine-tune on NYUv2 with depth-CP would
demonstrate that the composite does not harm downstream metrics, but
requires model weights, a dataset story, and a CI pathway the repository
does not currently carry. The composite's determinism is asserted by the
analytical goldens (monotonicity + metric rescale) plus the 200-seed
forward-gate snapshot; an end-to-end AbsRel regression is future
integration-test work.

### 10. Test strategy summary

- Analytical golden (monotonicity):
  `tests/fixtures/synthetic/depth_overlap.py` hand-constructs a
  `(target, source, expected)` triple with two planar surfaces at known
  depths (`d_tgt=2.0, d_src=1.0`); expected output bitwise equals
  `torch.where(m_eff, d_src, d_tgt)`.
- Analytical golden (intrinsics rescale):
  `tests/fixtures/synthetic/depth_intrinsics.py` with
  `fx_t=1000, fx_s=500` (isotropic) — output depth inside the paste
  region equals `2 * d_src`.
- Hypothesis fuzz: `dense_sample_strategy({Modality.IMAGE,
  Modality.INSTANCE, Modality.DEPTH, Modality.NORMALS})` at
  `max_examples=200`; all five invariant bodies called post-transform.
- Forward-gate snapshot: `tests/fixtures/depth_baseline.pt` generated
  via `scripts/gen_depth_baseline.py` on CPU at W4 HEAD; never
  regenerated (same policy as ADR-0005 §4). CUDA is skipped because
  `_effective_mask`'s `torch.where` + mask reductions are CPU-deterministic
  but not strictly guaranteed on CUDA.
- Invariant-matrix flip: five
  `InvariantRow(Modality.DEPTH|NORMALS, …, xfail=True)` entries in
  `tests/test_invariant_matrix.py:201-205` flip to passing.
- W2 + W3 parity regression canaries:
  `tests/test_dense_composite_parity.py` and
  `tests/test_panoptic_paste_parity.py` stay bitwise — emission of
  depth / depth_valid / normals is conditional on at least one input
  carrying them, so instance-only and panoptic-only paths are
  untouched.

## Consequences

- `DenseSample` gains a `metric_depth: bool = False` field; ADR-0001
  §(iii) table updated. `test_public_surface.py` remains green because
  the field has a default and `DenseSample` is constructed through
  `__init__` with keyword args.
- `DenseComposite.forward` now emits `depth`, `depth_valid`, and
  `normals` iff at least one input carries them. Instance-only and
  panoptic-only paths are unchanged.
- `DepthAwarePaste`, `DepthAwarePasteConfig` stay under
  `segpaste._internal`. Promotion requires an ADR-0007 amendment and an
  `_EXPECTED_PUBLIC_API` entry per ADR-0001 Part (i).
- ADR-0001 §(ii) validity-join wording is amended from AND-everywhere
  to the piecewise formula. `assert_depth_validity_join` signature
  changes (adds `effective_paste_mask` argument).
- `assert_depth_metric_intrinsics_rescale` signature is unchanged, but
  the ratio is computed as `sqrt(fx_t*fy_t) / sqrt(fx_s*fy_s)` instead
  of `fx_t / fx_s`. Tests carrying isotropic intrinsics are unaffected.
- `benchmarks/_fixture.py` grows `with_depth`, `with_normals`, and
  `metric_depth` parameters; `benchmarks/bench_depth_paste.py` lands
  alongside `bench_panoptic_paste.py`.
- ADR-0005 §5's private-until-validated policy moves one step closer to
  a follow-up promotion ADR: W4 exercises `DenseComposite` on two more
  modalities (depth + normals), with a wrapper-level h-flip
  preprocessing step that is orthogonal to the composite itself.

## Alternatives considered

- **`metric_depth` on `DepthAwarePasteConfig`.** Discarded: per-composite
  state forces users to construct two wrappers to mix metric and affine
  samples in a single dataloader. The flag is intrinsically per-sample —
  it states whether the tensor values are in meters or dimensionless
  relative depth — and pairs with `camera_intrinsics`, which is already
  per-sample. The amendment to ADR-0001 §(iii) is small (one field,
  default `False`).
- **Computing `min(d_src, d_tgt)` explicitly inside `_composite_depth`.**
  Discarded: redundant with `_effective_mask`, which already encodes
  the z-test. The `torch.where(m_eff, d_src, d_tgt)` formula is
  provably identical on `M_src ∩ placement`, one pass shorter, and
  mirrors the other composite branches.
- **ADR-0001 validity-join wording kept as-is.** Discarded: the
  AND-everywhere formula is wrong under a paste operation (lets the
  source's invalid region invalidate an un-touched target pixel). The
  bug would have surfaced the first time a monocular-depth user pasted
  a source with a larger invalid region than the target; W4's
  piecewise semantics is the correct fix.
- **Extend `PlacementSampler` with a `flip` parameter.** Discarded:
  more principled long-term but touches W3-shipped infra. The `flip`
  parameter would have to be threaded through `_place_things` in
  `PanopticPaste` and any future wrapper, and the W3 parity snapshot
  would need regeneration. Wrapper-level preprocessing isolates the
  change. If future wrappers need geometric augmentation beyond h-flip,
  a follow-up ADR can promote the extension.
- **Introduce a `BlendMode` enum.** Discarded: `Literal["alpha"]` via
  pydantic already enforces the restriction at construction. Adding an
  enum would duplicate ADR-0001's reserved-names list and require a
  surface-change amendment. The existing
  `CopyPasteConfig.blend_mode: Literal["alpha"]` pattern is reused.
- **Ray-rectified normal transport under translation.** Discarded per
  source report §8: the geometry is nontrivial (requires back-projection
  through `K` and forward-projection through `K`), per-pixel, and
  numerically delicate. P1 ships h-flip only; a follow-up ADR can add
  translation transport once a concrete downstream use case exists.
- **Runtime validity-join check in release.** Discarded: the composite
  construction is correct by design. A post-hoc assertion on every
  `transform` burns a full-frame reduction. Debug mode keeps the check
  as a fuzz-test harness opt-in.
