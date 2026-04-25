# ADR-0008 — `BatchCopyPaste`: GPU-resident kernel and `torch.compile` cleanliness

|            |                                                                                         |
| ---------- | --------------------------------------------------------------------------------------- |
| Number     | 0008                                                                                    |
| Title      | GPU-resident batched copy-paste kernel; deletion of the CPU path; compile-clean gate    |
| Status     | Accepted                                                                                |
| Author     | @NoeFontana                                                                             |
| Created    | 2026-04-23                                                                              |
| Updated    | 2026-04-23                                                                              |
| Tag        | `ADR-0008`                                                                              |
| Relates-to | [ADR-0002](0002-performance-baseline.md) Part (iv); [ADR-0003](0003-hard-deprecation-stance.md); [ADR-0004](0004-batched-dense-sample.md); [ADR-0005](0005-dense-composite.md); [ADR-0006](0006-panoptic-paste.md); [ADR-0007](0007-depth-aware-paste.md) |
| Amends     | [ADR-0002](0002-performance-baseline.md) Part (iv) → Part (v) (GPU throughput lane)     |

## Context

[ADR-0002](0002-performance-baseline.md) Part (i) locks the CPU baseline at
`141.2 ms` median per `CopyPasteCollator.__call__` (batch=8, `512²`,
`k ∼ U{1..5}`, `IQR/median ≈ 30%`) on `ubuntu-latest`. Phase 1's exit
criterion is a ≥2× throughput improvement, and Part (iv) explicitly
defers the GPU/CUDA lane to this workstream.

The four CPU wrappers that grew out of W1–W4 —
`InstancePaste` (ADR-0005), `PanopticPaste` (ADR-0006),
`DepthAwarePaste` (ADR-0007), and `ClassMix` — all share the same
item-dependent control-flow shape:

- `random.randint` / `random.sample` per image for source count and
  source selection (e.g. `instance_paste.py:69-77`,
  `classmix.py:64-108`);
- `.item()` calls to extract placement scalars
  (`placement.py:129, 134, 151-154, 184`;
  `composite.py:171`);
- `for i in range(int(src_masks.shape[0]))` loops in the stamp step
  (`instance_paste.py:101-144`,
  `panoptic_paste.py:121-142`).

None of this can trace through `torch.compile(fullgraph=True)` — every
`.item()` forces a graph break, and the Python-level RNG is non-dynamic.
`CopyPasteCollator` is a CPU-only entry point: it consumes
`list[DenseSample]` and produces `BatchedDenseSample` with intra-batch
source sampling, and cannot run on GPU-resident tensors at all.

W5 (workstream `M4` in the roadmap) closes that gap. The goal is a
single `nn.Module` entry point that:

1. Subsumes the four wrappers into one graph-compilable forward, exposing
   the same per-modality semantics pinned in ADRs 0005–0007.
2. Exposes its forward to Lightning's `on_after_batch_transfer` hook or
   as the first op of a `nn.Sequential` compile unit.
3. Consumes a fully-padded batched container so the compiled region
   carries no Python lists.

The pre-deletion CPU path is retained as the reference for a statistical-equivalence
gate (§6 below); after the soft-report window closes, the gate hardens
and the CPU path's parity fixtures are no longer a regression net.

## Decision

Land `BatchCopyPaste(nn.Module)` under
`src/segpaste/augmentation/batch_copy_paste.py`, a `PaddedBatchedDenseSample`
sibling container under `src/segpaste/types/`, and supporting
GPU-resident primitives under `src/segpaste/_internal/gpu/`. In the same
commit, hard-delete `CopyPasteCollator`, the four CPU wrappers
(`InstancePaste`, `PanopticPaste`, `DepthAwarePaste`, `ClassMix`),
`CopyPasteAugmentation`, both `placement.py` modules, the four
`*_baseline.pt` parity fixtures, the four parity tests, and the four
per-wrapper CPU benchmarks. No soft-deprecation shims; the pre-1.0
free-break window ([ADR-0003](0003-hard-deprecation-stance.md)) closes
by using it.

### 1. Scope: all four wrappers into one `nn.Module`

`BatchCopyPaste.forward(padded: PaddedBatchedDenseSample) ->
PaddedBatchedDenseSample` carries the instance, panoptic, depth+normals,
and class-mix semantics under one compilable graph. A single sampled
`(scale, translate, hflip)` tuple per paste is applied to every channel
group via one `grid_sample` call per group — the per-channel parameter
propagation that guarantees image, masks, depth, and normals stay
geometrically consistent. That consistency is what the four-wrapper
split cannot give.

`BatchCopyPasteConfig` is a frozen pydantic `BaseModel` with
`extra="forbid"`. It carries per-modality gate switches
(`emit_instance`, `emit_panoptic`, `emit_depth`, `emit_classmix`), the
shared `blend_mode: Literal["alpha"]` ([ADR-0001](0001-dense-sample.md)
`blend_mode` tightening), and the numeric caps
(`max_instances`, `max_attempts`, `tile_size`).

### 2. `PaddedBatchedDenseSample`: padded sibling, not a replacement

[ADR-0004](0004-batched-dense-sample.md) established `BatchedDenseSample`
with intentionally ragged instance-side fields
(`boxes: list[BoundingBoxes]`, `labels: list[Tensor]`,
`instance_masks: list[InstanceMask]`, `instance_ids: list[Tensor]`,
`camera_intrinsics: list[CameraIntrinsics]`). Ragged is correct for the
CPU path and for dataloader output; it is a `torch.compile` hazard for
the GPU path.

W5 adds a sibling `PaddedBatchedDenseSample` at
`src/segpaste/types/padded_batched_dense_sample.py`:

| Field | Type | Shape |
| --- | --- | --- |
| `images` | `tv_tensors.Image` | `[B, C, H, W]` (channels_last) |
| `boxes` | `torch.Tensor` | `[B, K, 4]` float32 xyxy |
| `labels` | `torch.Tensor` | `[B, K]` int64 |
| `instance_masks` | `torch.Tensor` | `[B, K, H, W]` bool |
| `instance_ids` | `torch.Tensor` | `[B, K]` int32 |
| `instance_valid` | `torch.Tensor` | `[B, K]` bool (padding mask) |
| `semantic_maps` | `SemanticMap \| None` | `[B, H, W]` int64 |
| `panoptic_maps` | `PanopticMap \| None` | `[B, H, W]` int64 |
| `depth` | `torch.Tensor \| None` | `[B, 1, H, W]` float32 (channels_last) |
| `depth_valid` | `torch.Tensor \| None` | `[B, 1, H, W]` bool |
| `normals` | `torch.Tensor \| None` | `[B, 3, H, W]` float32 (channels_last) |
| `camera_intrinsics` | `torch.Tensor \| None` | `[B, 4]` float32 (`fx, fy, cx, cy`) |

`instance_valid` is the per-row padding mask. Invalid rows are zeroed
post-construction and **every write in `BatchCopyPaste.forward` is
gated on `instance_valid`** — an invalid row can never leave a pixel
in the composite.

`BatchedDenseSample.to_padded(max_instances: int) ->
PaddedBatchedDenseSample` and `PaddedBatchedDenseSample.to_batched() ->
BatchedDenseSample` form a roundtrip. `to_padded` truncates rows beyond
`max_instances` and raises if any sample carries more instances than
`max_instances`; the callsite (dataloader assembly or training loop)
picks `max_instances` with visibility on its dataset.

ADR-0004's field table and semantics are unamended; the padded form is
a view, not a replacement. `__post_init__` validation runs under
`@skip_if_compiling` per the [ADR-0004](0004-batched-dense-sample.md)
convention.

### 3. Intra-batch source sampling; `InstanceBank` deferred

Source instance selection is a `torch.multinomial` over the flattened
`[B*K]` instance index, masked to keep sources out of the target's own
row. This matches today's `CopyPasteCollator` semantics exactly — sources
come from other samples in the same batch — so the KS-equivalence gate
(§6) compares like-with-like.

A persistent `InstanceBank` (pycocotools RLE masks off-GPU,
class-balanced sampler) is a meaningful upgrade for LVIS-scale training
where `B=8` provides thin class diversity, but it introduces a
dataset-prep step and a new public surface. It is deferred to the
successor ADR (targeted `ADR-0009`). W5 pins the interface assumption:
`BatchCopyPaste` accepts a `source_pool` argument that currently defaults
to `None` (intra-batch) and will later accept an `InstanceBank` instance
without changing the config surface.

### 4. Tile compositing at 512² with mirrored edges

At `B=8, 2048²` (Cityscapes panoptic), the stacked
`PaddedBatchedDenseSample` occupies `~3 GB` for the image tensor alone;
with `K=8` instance masks per image (`bool [B, K, H, W]`), one full-frame
composite pass materializes temporaries that push peak GPU memory well
above the 40 GB A100-SXM budget. Tile compositing at `512²` with
mirrored edges bounds peak memory per pass.

The tile iterator at `src/segpaste/_internal/gpu/tile_composite.py`
calls `DenseComposite.forward` (unchanged from ADR-0005) per tile with
clipped paste masks; the mirrored edge ensures `grid_sample` outputs do
not see padding discontinuities at tile seams. Reconciliation is
`torch.where` over the tile-boundary pixels with the validity mask.

The tile size is fixed at `512` for W5. Making it configurable has no
client need at present; changing it is an additive patch under this ADR.

Tile correctness is anchored by an explicit test: at `tile=img_size`,
the reconciled output bitwise equals the full-frame
`DenseComposite.forward` result.

### 5. Per-channel `grid_sample` propagator

`src/segpaste/_internal/gpu/affine_propagate.py::apply_affine(padded,
scale, translate, hflip)` generates one sampling grid and calls
`grid_sample` per channel group:

- Image: `mode='bilinear'`, `align_corners=False`.
- Instance masks, `semantic_map`, `panoptic_map`: `mode='nearest'`,
  `align_corners=False` — preserves integer labels, and the
  cardinality-`{0, 1}` invariant for bool masks is asserted in
  `tests/test_affine_propagate.py`.
- Depth: `mode='bilinear'`; invalid pixels are filled explicitly to
  `nan` before sampling, re-interpreted as invalid post-sample via
  `depth_valid`. The `depth_valid` tensor itself samples at `nearest`.
- Normals: `mode='bilinear'`; when the `hflip` branch fires,
  `n_x = -n_x` is applied on the output, preserving the right-down-forward
  camera-frame convention pinned in [ADR-0007](0007-depth-aware-paste.md)
  §7.

Translation is integer-pixel by construction (sampled from integer
grids); the `align_corners=False` + integer-translation convention
prevents `grid_sample`'s nearest-mode from flipping boundary pixels.

### 6. KS statistical-equivalence gate: soft-report for 30 days, then harden

Bitwise CPU↔GPU parity is not required; numerical drift from
`grid_sample` vs. integer cropping, and RNG-device drift
(`torch.randint` on CUDA is deterministic but not seed-identical to
CPU), would make such a gate falsely fail.

The equivalence contract is per-modality KS distance on three
histograms:

1. **Paste area** (pixels per pasted instance).
2. **Number of pastes per image** (per sample of the batch).
3. **Per-class paste count** (top-20 classes by paste frequency).

At commit `C6`, `scripts/gen_ks_snapshot.py` runs through the
pre-deletion CPU wrappers at `n=1000` draws per modality and writes
`tests/fixtures/ks_snapshot.pt`. The reference is immutable and
committed alongside the deletion commit (`C7`) — the pre-deletion CPU
behavior is frozen into the fixture.

`tests/test_batch_copy_paste_ks.py` computes
`scipy.stats.ks_2samp(cpu_hist, gpu_hist)` at `n=1000` per modality
for each of the three histograms and writes the full distance table
to a CI artifact. For 30 days the test asserts nothing — it records
only. After the soft-report window closes, this ADR is amended to pin
a hard threshold (targeted: `KS ≤ 0.05`, two-sided, `α=0.01` per
modality-histogram pair). The threshold pin is a one-line amendment,
not a new ADR, because Part (iii) of ADR-0002's acceptance framework
applies mutatis mutandis.

### 7. Compile-clean CI gate

`scripts/compile_explain.py` runs `torch._dynamo.explain` on a CPU
trace of `BatchCopyPaste.forward` against a fixture
`PaddedBatchedDenseSample`, captures the graph-break reason list, and
diffs against `scripts/compile_allowlist.txt`. The allow-list is empty
at M4 and additions require this ADR to be amended (the reason and
the offending operation are pinned into the file alongside the
allow-list entry).

The gate runs on CPU because `torch._dynamo.explain` does not require
a GPU runner — dynamo's trace operates on FakeTensor. This lets every
PR enforce compile-cleanliness without a self-hosted A100 runner. The
actual A100 throughput measurement is a separate, nightly,
`workflow_dispatch`-only bench (§9).

`BatchCopyPaste` is authored to `fullgraph=True` standards:

- No `.item()` anywhere in the forward path.
- No Python `random` calls; all sampling uses `torch.randint` /
  `torch.multinomial` with an explicit `torch.Generator` argument.
- No Python-level `if tensor_value` branches; all branching is
  `torch.where`.
- No `tuple(tensor.tolist())` patterns; shape-dependent Python control
  flow is replaced by tensor-dimension indexing.

`CopyPasteConfig.blend_mode: Literal["alpha"]` is preserved on
`BatchCopyPasteConfig`. No `BlendMode` enum is introduced
([ADR-0007](0007-depth-aware-paste.md) §6).

### 8. GPU CI policy and the ADR-0002 Part (iv) → Part (v) amendment

[ADR-0002](0002-performance-baseline.md) Part (iv) defers the GPU lane
and the A100 runner to P0.D. W5 discharges the deferral in two parts:

- **Compile-clean on every PR.** No GPU required; runs on
  `ubuntu-latest`.
- **Throughput bench on nightly `workflow_dispatch` only.** Does require
  an A100 SXM runner; runs only when the maintainer triggers it
  manually. PR-level GPU gating is deferred until a persistent self-hosted
  runner is provisioned — tracked in the ADR-0002 amendment as Part (v).

The full Part (v) text is appended to ADR-0002 in the same commit as
this ADR.

### 9. Deletion manifest, single commit

At commit `C7`, a single commit adds `BatchCopyPaste` and deletes every
superseded symbol:

- **Public surface:** `CopyPasteCollator` is removed from
  `segpaste.__all__` and
  `tests/test_public_surface.py::_EXPECTED_PUBLIC_API`.
  `BatchCopyPaste` and `PaddedBatchedDenseSample` are added in the same
  diff. `BatchCopyPaste.from_dataloader(loader, max_instances: int)`
  is the documented migration helper.
- **Internal:** `src/segpaste/_internal/instance_paste.py`,
  `panoptic_paste.py`, `depth_paste.py`, `classmix.py`, `placement.py`,
  and `src/segpaste/processing/placement.py` are deleted outright.
  `src/segpaste/augmentation/copy_paste.py` (`CopyPasteAugmentation`)
  and `src/segpaste/augmentation/torchvision.py` (`CopyPasteCollator`)
  are deleted outright.
- **Fixtures:** `tests/fixtures/composite_baseline.pt`,
  `depth_baseline.pt`, `panoptic_baseline.pt` are deleted.
- **Tests:** `tests/test_dense_composite_parity.py`,
  `test_depth_paste_parity.py`, `test_panoptic_paste_parity.py`,
  `test_copy_paste.py`, `test_copy_paste_fuzz.py`,
  `test_placement_fuzz.py`, `test_depth_paste.py`,
  `test_panoptic_paste.py`, `test_classmix.py` are deleted.
- **Scripts / benchmarks:** `scripts/gen_composite_baseline.py`,
  `gen_depth_baseline.py`, `gen_panoptic_baseline.py`;
  `benchmarks/bench_copy_paste.py`, `bench_panoptic_paste.py`,
  `bench_depth_paste.py`, `bench_classmix.py`, `benchmarks/_fixture.py`
  are deleted.

The deletion is one commit because partial-migration state would leave
`main` green-but-wrong: users importing `CopyPasteCollator` would either
succeed (pre-deletion commits) or fail (post-deletion commits); there is
no intermediate contract worth shipping. `BatchCopyPaste.from_dataloader`
covers the migration ergonomics. This is the canonical hard-deprecation
example cited in the ADR-0003 amendment.

### 10. `torch==2.8.*` pin

`pyproject.toml` narrows its `torch>=2.8` dependency to `torch==2.8.*`.
The compile-clean allow-list is a graph-break-reason string diff, and
those strings are `torch._dynamo` internal APIs that change across
minor versions. Pinning the minor keeps the allow-list stable. Upgrading
to torch 2.9 becomes a dedicated PR that re-validates the compile report
against the new minor — tracked as a follow-up, not blocked on W5.

## Consequences

- **Public surface delta.** `segpaste.__all__` gains
  `BatchCopyPaste`, `PaddedBatchedDenseSample`; loses `CopyPasteCollator`.
  `segpaste.augmentation.__all__` gains `BatchCopyPaste`; loses
  `CopyPasteCollator`. `segpaste.types.__all__` gains
  `PaddedBatchedDenseSample`. `tests/test_public_surface.py` is updated
  in the same commit.
- **Loss of per-wrapper CPU regression nets.** The four `*_baseline.pt`
  fixtures are deleted. `ks_snapshot.pt` is the new ground truth. Users
  who want per-wrapper CPU parity can pin `segpaste<0.10`.
- **New `_internal` modules.** `src/segpaste/_internal/gpu/` lands with
  `batched_placement.py`, `affine_propagate.py`, `tile_composite.py`.
  Promotion to `segpaste.__all__` requires a follow-up ADR per
  [ADR-0005](0005-dense-composite.md) §5.
- **CI shape.** `.github/workflows/ci.yml` gains a `compile-clean` step;
  `.github/workflows/bench-gpu.yml` lands as `workflow_dispatch`-only.
- **`DenseComposite` unchanged.** The composite (ADR-0005) remains the
  pixelwise-where primitive; the tile iterator consumes it per-tile.
  ADRs 0005–0007 are referenced, not amended.
- **Invariant matrix unchanged.** `tests/test_invariant_matrix.py`
  remains green — invariant bodies are not touched; only callers change.
- **`BatchedDenseSample` gains roundtrip methods.** `to_padded` /
  `from_padded`; no field changes; ADR-0004 is referenced, not amended.
- **`CHANGELOG.md` `### Removed` section** for the next minor release
  lists the five public/internal symbols, four fixtures, four parity
  tests, four CPU benches, and three baseline-generation scripts
  deleted at C7.

## Alternatives considered

- **Instance-only scope at M4.** Discarded after user direction:
  landing all four modalities in one `BatchCopyPaste` matches the
  per-channel grid_sample propagation goal directly. Splitting the
  modalities across four separate GPU modules reproduces the
  four-wrapper CPU shape; the whole point of the merge is that
  geometric consistency across modalities requires a single sampling
  grid.
- **Deprecate-with-warning CPU path; hard-delete in successor ADR.**
  Discarded per ADR-0003 ("using the pre-1.0 free-break window to
  actually remove code is cheaper than using it to build more
  scaffolding"). A 30-day soft-report window for numerical
  equivalence (§6) is the protection; a deprecation-warning window for
  surface stability is not.
- **Keep the four CPU wrappers as `_internal` parity-gate anchors.**
  Discarded: they carry no ongoing value once `ks_snapshot.pt` is
  committed, and their presence would mean the KS reference path still
  runs on every PR. The fixtures already freeze the pre-deletion
  behavior.
- **InstanceBank in W5.** Discarded: a persistent class-balanced bank
  introduces a dataset-prep step, an RLE storage decision, and a new
  public class. W5 is already a maximally-scoped deletion + GPU port;
  the bank is ADR-0009 material.
- **Hard KS gate from day one.** Discarded: CPU/GPU RNG divergence +
  `grid_sample` vs. integer-crop drift make threshold selection a
  measurement question, not a design question. The 30-day soft-report
  window is calibration; the hard threshold is a one-line amendment.
- **Full-frame composite without tiling.** Discarded at Cityscapes
  panoptic batch-8 `2048²`: memory math puts peak well above the
  40 GB A100-SXM budget. Tile compositing is not a future-proofing
  choice; it is the current-scale necessity.
- **Configurable tile size.** Discarded: `512` is a single number with
  no current tuning need. Making it configurable now broadens the
  compile-clean allow-list surface (different `tile_size` values will
  produce different trace shapes) without a client need.
- **Bitwise CPU-vs-GPU parity under matched seeds.** Discarded: not
  achievable without cuDNN/cuBLAS determinism guarantees this project
  does not want to inherit. Statistical equivalence (§6) is the
  defensible claim.
- **Compile-clean allow-list managed in a separate docs file.**
  Discarded: the allow-list is load-bearing CI state; it needs to live
  next to the script that reads it, not in documentation that could
  drift.
