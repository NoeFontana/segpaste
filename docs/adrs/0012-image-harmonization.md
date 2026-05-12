# ADR-0012 — Torch-native image harmonization (Reinhard / multi-band / Poisson)

|            |                                                                                                                                                       |
| ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| Number     | 0012                                                                                                                                                  |
| Title      | `HarmonizeConfig` + `ImageHarmonizer`: torch-native image harmonization between `AffinePropagator` and `TileCompositor`                               |
| Status     | Accepted                                                                                                                                              |
| Author     | @NoeFontana                                                                                                                                           |
| Created    | 2026-05-12                                                                                                                                            |
| Tag        | `ADR-0012`                                                                                                                                            |
| Relates-to | [ADR-0001](0001-dense-sample.md) Part (i) (public surface, additive Literal widening); [ADR-0007](0007-depth-aware-paste.md) §7 (alpha-only Literal); [ADR-0008](0008-batch-copy-paste.md) §D7 (compile-clean allow-list) |
| Amends     | [ADR-0001](0001-dense-sample.md) Part (i) — closes the v0.2.0 reservation note for `blend_mode` `"gaussian"` / `"poisson"`                            |

## Context

`CHANGELOG.md` v0.2.0 (lines 86–88) and ADR-0001 §(i) lines 113–118 leave a
note: `CopyPasteConfig.blend_mode` had reserved-but-unwired `"gaussian"` and
`"poisson"` literals, tightened to `["alpha"]` for v0.2.0 with the explicit
intent to re-introduce blend modes "via additive ADR" once their wiring
landed. v0.3.0 hard-deleted the entire CPU augmentation stack including
`CopyPasteConfig` itself, but the design intent — that `BatchCopyPaste`
should eventually offer harmonized composites — carried forward.

A pre-A3 proposal layered three external tiers (OpenCV `seamlessClone`, the
Harmonizer model, DCCF) to satisfy this. Costs of that path: a 60+ MB OpenCV
wheel with an image-codec dependency tree for one function, two model-weights
distributions (each with its own license review and `check_no_binaries.py`
entry), a post-forward CPU stage that breaks the compiled forward, and
allow-list amendments for whatever graph breaks those tiers introduce.

Each of those costs is unjustified by the underlying problem. The three
classical methods that account for ~90% of what `seamlessClone` provides
perceptually — Reinhard 2001, Burt-Adelson 1983, Pérez et al. 2003 — are
each implementable in pure torch in 80–180 LOC, fit inside the compiled
forward, and require no new dependencies, no model weights, no extras, no
allow-list churn. This ADR ships all three.

## Decision

A new nested config and `nn.Module`, both under `_internal`:

```python
class HarmonizeConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    mode: Literal["reinhard", "multiband", "poisson"] = "multiband"
    prob: float = Field(default=0.0, ge=0.0, le=1.0)
    pyramid_levels: int = Field(default=5, gt=0)

class BatchCopyPasteConfig(BaseModel):
    ...
    harmonize: HarmonizeConfig = Field(default_factory=HarmonizeConfig)
```

Default `prob=0.0` makes the harmonizer a graph-clean identity; v0.3.0
behavior is byte-for-byte preserved when the field is unset. Mode default
`"multiband"` is the recommended choice when a user opts in.

### Pipeline placement

`ImageHarmonizer` is a new step in `BatchCopyPaste.forward` *between*
`AffinePropagator` and `TileCompositor`:

```python
warped = self.propagator(padded, source_view, placement)
paste_mask = self._paste_mask(warped, placement)
warped = self.harmonizer(padded, warped, paste_mask, generator)  # ADR-0012
composited = self.compositor(padded, warped, paste_mask)
```

The harmonizer operates on the **full warped image**, not per-tile. Multi-
band pyramids and DST solves require global support; per-tile operation
would either reintroduce seams at tile boundaries or require halo exchange.
The full-image cost is bounded (multi-band is five small convs per pyramid
level; DST is two batched matmuls per direction).

The downstream `TileCompositor` is unchanged. Its per-tile
`torch.where(m_eff, src, tgt)` still runs against the harmonized warped
image, so the single-pass-equals-tiled invariant in
`tests/test_tile_composite.py` is preserved (the alpha-where math is
unchanged; only the value of `src` shifts).

Z-test rejected pixels (closer target depth) and pixels outside the paste
mask receive the target. Their harmonized values are computed and unused —
the deliberate cost of keeping harmonization global.

### Per-image bernoulli mixing

When `prob > 0`, both arms always run, then a per-image bernoulli draw
selects between the harmonized and un-harmonized warped image:

```python
harmonized = self._<mode>(warped_img, target_img, paste_mask)
bern = torch.bernoulli(torch.full((B,), prob, ...), generator=...)
out = where(bern.view(B, 1, 1, 1), harmonized, warped_img)
```

This exposes the model to both composites under a curriculum dial. Mode
dispatch is by Python `if/elif` on `self.config.mode`, resolved at trace
time so `torch.compile` specializes one branch per module instance.

### The three modes

**Reinhard 2001 (statistical color transfer, ~90 LOC).** Convert RGB → LMS
via the fixed 3×3 matrix from the original paper, take `log10`, project to
the perceptually-decorrelated Lαβ basis. Per-image, per-channel statistics
`(μ_s, σ_s)` are computed over the warped image *inside* the paste mask;
`(μ_t, σ_t)` over the target image *outside* the paste mask. Pixelwise
`x' = (σ_t / σ_s)(x - μ_s) + μ_t`; inverse-transform back to RGB; clamp to
`[0, 1]`. Empty-mask handling: the denominator is clamped to ``1`` and the
variance to `1e-12` so the per-image computation stays graph-clean and
returns a well-defined identity when a sample has no paste.

**Multi-band Burt & Adelson 1983 (~120 LOC).** Build Gaussian pyramids of
`warped`, `target`, and `paste_mask.float()` over `K=5` levels (capped at
`floor(log2(min(H, W))) - 1`); build Laplacian pyramids of `warped` and
`target`; blend each Laplacian level by the corresponding mask Gaussian
level — `L_out^k = G_M^k * L_s^k + (1 - G_M^k) * L_t^k`; collapse the
blended Laplacian top-down. The 5×5 binomial kernel
`[1, 4, 6, 4, 1] / 16` is registered as a non-persistent buffer and used
via grouped conv. Up-sampling uses
`F.interpolate(mode="bilinear", align_corners=False)`. Coarse mask levels
(large support) handle low-frequency color/illumination mismatch; fine mask
levels preserve the silhouette.

**Poisson via DST (~110 LOC).** Pérez et al. 2003 seamless cloning in the
substitution `u' = u - tgt`, so the boundary condition is homogeneous
(``u' = 0`` on the image edge) — exactly what the type-I DST diagonalizes.
Forcing field `f = (Δsrc - Δtgt) · mask`; inside the paste mask this
enforces the source's gradient field, outside the mask `f = 0` so the
solution relaxes back to `tgt`. The 1-D discrete Laplacian eigenvalues are
`λ_i = -4 sin²(π · i / (2 · (N + 1)))`; we solve
`DST{u'}_ij = -DST{f}_ij / |λ_ij|` then inverse-DST. The DST-I is computed
as the orthonormal matrix `M[i,j] = √(2/(N+1)) sin(π(i+1)(j+1)/(N+1))` and
applied as two batched matmuls per direction (DST-I is its own inverse for
the orthonormal scaling, so forward and inverse share `M`). Cost is
`O(B · C · H · W · (H + W))` per direction.

The full-image formulation differs from the per-bbox formulation in the
original Pérez paper: with image-edge boundary conditions, the global
solve may bleed very-low-frequency tint into the region just outside the
paste mask. For typical copy-paste scales (paste rect ≪ image) this is
bounded, and the downstream alpha-where snaps the unmasked region back to
the target value anyway. The benefit is a single fixed-shape DST per batch
that is straightforwardly compile-clean — no per-paste variable-bbox
loops.

## Consequences

### Compile-clean

`scripts/compile_allowlist.txt` stays empty. Every op used by the three
modes traces clean under `torch.compile(fullgraph=True)`:

- `torch.bernoulli` with explicit `generator` and pre-computed full-tensor
  argument: graph-clean.
- `F.conv2d` with grouped fixed-kernel convs (binomial / Laplacian
  stencil): graph-clean.
- `F.interpolate(mode="bilinear", align_corners=False)`: graph-clean.
- `torch.einsum("cd,bdhw->bchw", ...)` with fixed-matrix buffers:
  graph-clean.
- `torch.matmul`, `torch.log10`, `torch.pow(10.0, ...)`, `torch.sin`,
  `torch.arange`, `.clamp`, `.expand().contiguous()`: all graph-clean.
- The pyramid level cap and the mode dispatch are Python ints / strings
  resolved at trace time, so dynamo specializes one path per module
  instance.

The pre-merge gate is `tests/test_compile_clean.py`'s new
`test_harmonize_modes_have_no_disallowed_breaks` parametrized over the
three modes at `prob=1.0` — the harmonize branch must be exercised, not
the identity fast path.

### Public surface

Zero changes to `segpaste.__all__` and `_EXPECTED_PUBLIC_API`.
`HarmonizeConfig` follows the sibling pattern of `BatchedPlacementConfig`,
`TileCompositorConfig`, `PanopticPasteConfig` — reachable through
`BatchCopyPasteConfig.harmonize` but not re-exported at top level.
`ImageHarmonizer` is private under `_internal`.

The type-system change (the new `Literal["reinhard", "multiband",
"poisson"]` on `HarmonizeConfig.mode`) is additive per ADR-0001 Part (i):
no existing config field changes shape or default. Constructing
`BatchCopyPasteConfig()` produces the same module behavior as v0.3.0.

### Dependencies

No additions to `pyproject.toml`. No new optional-dependency group; no
`[harmonize]` extras (one was never created). No model weights, no licensed
binaries, no impact on `scripts/check_no_binaries.py`.

### Dead code removal

`src/segpaste/processing/` is deleted in this ADR's PR. The subpackage
held pre-v0.3.0 per-image (`[C, H, W]` shape) blending utilities
(`alpha_blend`, `gaussian_blend`, `blend_with_mode`,
`create_smooth_mask_border`) plus `boxes_to_masks` / `compute_mask_area`
helpers, and was orphaned from the GPU pipeline since v0.3.0. The single
remaining call site (`_internal/invariants/instance.py`'s use of
`compute_mask_area`) is inlined to `masks.to(torch.bool).sum(dim=(1, 2))`.
Per ADR-0003's hard-deprecation stance, the subpackage is removed outright
rather than soft-deprecated.

### Tests

`tests/test_harmonize.py` covers:

- `test_prob_zero_returns_warped_unchanged` per mode — fast-path identity.
- `test_prob_one_changes_warped` per mode — harmonize branch fires.
- `test_bernoulli_per_image_mixing` per mode — per-image bernoulli does
  populate both arms at `prob=0.5` with `B=32`.
- `test_reinhard_uniform_to_uniform_matches_target_color` — Reinhard pulls
  uniform-color source toward uniform-color target's mean.
- `test_multiband_seam_is_smoother_than_alpha` — multi-band reduces seam
  gradient magnitude vs. hard alpha at the paste boundary.
- `test_poisson_outside_mask_close_to_target` — Poisson solution stays
  near the target outside the paste mask (bounded global-solve bleed).
- `test_output_shape_dtype_preserved`, `test_output_in_unit_range` per
  mode — RGB stays clamped.
- `test_multiband_pyramid_caps_for_small_images` — the
  `floor(log2(min(H,W))) - 1` cap kicks in for sub-32px fixtures.
- `test_dst_matrix_is_orthonormal`,
  `test_dst_eigenvalues_strictly_positive` — math primitives.
- `test_config_rejects_unknown_mode`, `..._extra_fields`,
  `..._prob_out_of_range` — Pydantic validation.
- `tests/test_compile_clean.py::test_harmonize_modes_have_no_disallowed_breaks`
  — compile-clean regression per mode.

### Out of scope

External harmonization tiers — the `Harmonizer` model, DCCF, OpenCV
`seamlessClone` — are explicitly out of scope. If a downstream user
demonstrates a measurable gap that the in-graph torch-native methods can't
close, a follow-up ADR can revisit. Until then, the door stays closed.

The per-bbox formulation of Poisson cloning (DST on the bbox-of-mask
rectangle with Dirichlet conditions from the target's bbox boundary) is
also out of scope. The full-image formulation chosen here is the only
formulation that batches cleanly without per-image variable-bbox loops in
the compiled forward.

## Status

Accepted. Implementation lands in this PR.
