# ADR-0010 — Patch-aligned copy-paste for ViT backbones

|            |                                                                                                                                                                                          |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Number     | 0010                                                                                                                                                                                     |
| Title      | `pad_to_multiple` and `patch_aligned_paste`: canvas padding and placement snapping for ViT patch-token alignment (A2)                                                                    |
| Status     | Accepted                                                                                                                                                                                 |
| Author     | @NoeFontana                                                                                                                                                                              |
| Created    | 2026-05-10                                                                                                                                                                               |
| Tag        | `ADR-0010`                                                                                                                                                                               |
| Relates-to | [ADR-0008](0008-batch-copy-paste.md) §C3 (placement sampler), §C4 (affine propagator)                                                                                                    |
| Amends     | [ADR-0008](0008-batch-copy-paste.md) §C3 (extends `BatchedPlacement` with `src_valid_extent`); §C4 (`_build_grid` and `_transform_boxes` reflect about per-source pre-pad valid extent)  |

## Context

§7.7 of the report records an open question: HuggingFace
`Mask2FormerImageProcessor` defaults `size_divisor=32`, which silently breaks
DINOv2 ViT-L/14 (patch=14). Sub-patch paste boundaries inside a single ViT
token force the linear patch-embed to absorb a high-frequency intensity
discontinuity that the Gaussian α-mask only partially hides — the §2
"patch-tokenization × paste boundary" open question.

Without canvas pad-to-multiple and placement snapping, every
`BatchCopyPaste` paste lands at sub-patch granularity. The patch-embed
sees a token straddling a paste boundary and learns to associate the
discontinuity pattern with class transitions — a confound for the empirical
ablation slated for 1.0.0.

## Decision

Two new fields on `BatchedPlacementConfig` at
`src/segpaste/_internal/gpu/batched_placement.py`:

```python
pad_to_multiple: int | None = Field(default=None, gt=0)
patch_aligned_paste: bool = False
```

A `model_validator` rejects `patch_aligned_paste=True` without
`pad_to_multiple` set — the alignment guarantee requires the canvas itself
to be divisible by `p`.

### 1. Canvas padder

`pad_canvas_to_multiple` at `src/segpaste/_internal/gpu/pad_canvas.py`
right-and-bottom-pads every spatial field of a `PaddedBatchedDenseSample` by
`(pad_h, pad_w) = (-H % p, -W % p)`:

| Field             | Pad mode  | Fill            |
| ----------------- | --------- | --------------- |
| `images`          | reflect   | —               |
| `instance_masks`  | constant  | `False`         |
| `semantic_maps`   | constant  | `ignore_index`  |
| `panoptic_maps`   | constant  | `ignore_index`  |
| `depth`           | constant  | `0.0`           |
| `depth_valid`     | constant  | `False`         |
| `normals`         | constant  | `0.0`           |
| `padding_mask`    | constant  | `True`          |

`boxes`, `instance_valid`, and `camera_intrinsics` do not depend on the
canvas extent and are forwarded unchanged. `ignore_index` flows through
from `BatchCopyPasteConfig.panoptic.taxonomy.ignore_index` when panoptic
mode is active, else falls back to `255` (LSJ convention,
`augmentation/lsj.py:75`).

`(pad_h, pad_w) = (0, 0)` returns the input object unchanged — already-
divisible canvases skip the pad step entirely.

### 2. Placement snapping (`patch_aligned_paste`)

When `patch_aligned_paste` is `True`, `BatchedPlacementSampler.forward`:

1. **Discretizes scale**. Replaces the continuous-uniform draw on
   `scale_range` with a discrete-uniform draw on the grid
   `n · p / gcd(H, W)` for `n ∈ [⌈smin · gcd / p⌉, ⌊smax · gcd / p⌋]`.
   The grid is the smallest set of scales for which both `s · H` and
   `s · W` are integer multiples of `p` — i.e. the joint constraint
   `⌊s · H⌋ mod p = 0 ∧ ⌊s · W⌋ mod p = 0` holds exactly.
2. **Floor-snaps translates**. `ty ← ⌊ty / p⌋ · p`, same for `tx`. Both
   monotonically decrease, preserving the existing `fits` check.

If the discrete scale grid contains no point inside `scale_range` (e.g.
`gcd(518, 686) = 14, p = 14, quanta = 1.0` with `scale_range = (0.5, 0.6)`),
the sampler raises `RuntimeError` with a config-mismatch message. This is
a runtime check rather than config-time because canvas size is not known
until `forward`.

### 3. Hflip-via-`src_valid_extent`

Pre-A2, `BatchedPlacementSampler` and `AffinePropagator` reflected hflipped
content about the post-pad canvas centerline `(W - 1) / 2`. After A2's pad
step, that centerline is `pad_w / 2` to the right of the source content's
true centerline `(W_valid - 1) / 2`, so hflipped slots would shift content
into the pad band on one side and pull pad zeros into the visible region on
the other. (This was a pre-existing latent issue for any LSJ output smaller
than the target canvas; A2 makes it always-on.)

`BatchedPlacement` gains `src_valid_extent: Tensor | None` (`[B, 2]` per-
target carrying the source's pre-pad `(h_v, w_v)`). Sampler computes it as
`valid_extent[source_idx]` and uses `src_valid_extent[:, 1]` for the
`eff_x2` hflip term. `AffinePropagator._build_grid` and `_transform_boxes`
take `src_valid_extent` as a new parameter and reflect about
`src_w_valid - 1` instead of `w - 1`. Grid normalization still uses the
post-pad `(h, w)` because `grid_sample` reads from the post-pad source
tensor.

`src_valid_extent = None` falls back to the post-pad `(H, W)` broadcast —
preserves the legacy behavior for code paths that construct a
`BatchedPlacement` directly (e.g. unit tests).

## Consequences

### Compile-clean

`math.gcd` resolves at trace time (Python ints from
`padded.images.shape`); the discrete-uniform `torch.randint` and the
`torch.floor` snap on `ty`, `tx` are graph-clean. `F.pad` with mode
`reflect` and mode `constant` lowers without breaks. Allowlist at
`scripts/compile_allowlist.txt` remains empty; the compile-clean test
adds a second case exercising `patch_aligned_paste=True`.

### Caveats

1. **Sub-pixel slop near bottom-right edge.** The constraint
   `⌊s · H⌋ mod p = 0` aligns the rendered extent at `s · H`, but
   `grid_sample` indexes pixels `[0, H − 1]`, so the bottom-right paste
   boundary lands at `ty + s · (H − 1)` ≈ `ty + s · H − s`. Net misalignment
   is bounded by `s` pixels (fractional under `align_corners=False`). For
   `s ≤ 2` this is well below the `p`-pixel patch-embed token spacing.
2. **Coarse-quanta runtime guard.** Rectangular canvases can produce
   `quanta = p / gcd(H, W)` larger than the requested `scale_range` width.
   The sampler raises `RuntimeError` rather than silently degrading the
   scale distribution. Users with rectangular canvases should pick `p`
   such that `gcd(H, W) ≥ p` (e.g. pad to square first).
3. **Reflect-pad introduces synthetic edge content.** Constant-pad would
   match HuggingFace `Mask2FormerImageProcessor` and the LSJ convention but
   re-introduces the hard zero edge that A2 is trying to suppress. Reflect
   is chosen per the user-spec'd intent: pad the canvas with content
   continuous with the visible image so the patch-embed at the boundary
   does not see a sharp transition. Downstream consumers (Mask2Former
   pixel-mask path) gate the pad band off via `padding_mask`, so reflected
   pixels never enter the loss.

### Public surface

`BatchedPlacementConfig` and `BatchCopyPasteConfig` are not in
`segpaste.__all__`; the new fields and the `BatchedPlacement.src_valid_extent`
field are internal. No
`tests/test_public_surface.py` update
required.

### Tests

- Config validator coverage in `test_batched_placement.py`.
- Alignment invariant for `p ∈ {14, 16}` over square and rectangular
  canvases.
- `paste_valid` degradation property test (≥95% retention vs. unaligned
  baseline at the same seed).
- `tests/test_pad_canvas.py` per-modality fill assertions and fast-path
  identity.
- Hflip-with-pad regression in `test_affine_propagate.py`.
- Coarse-quanta runtime guard.
- Compile-clean re-verify with `patch_aligned_paste=True`.

## Status

Accepted. Implementation tracked under workstream A2 (2-week budget); the
empirical ablation tying paste-boundary patch alignment to ViT patch-embed
behavior is deferred to 1.0.0.
