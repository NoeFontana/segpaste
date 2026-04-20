# ADR-0001 — Dense-Sample Composites

|            |                                                      |
| ---------- | ---------------------------------------------------- |
| Number     | 0001                                                 |
| Title      | Dense-sample composites (instance, panoptic, semantic, depth, normals) |
| Status     | Accepted                                             |
| Author     | @NoeFontana                                          |
| Created    | 2026-04-20                                           |
| Updated    | 2026-04-20                                           |
| Tag        | `ADR-0001`                                           |

Every Phase 1 (P1) change that touches a symbol, invariant, or field pinned
here should reference `ADR-0001` in its commit message or PR description.
Superseding a decision means writing a new ADR that supersedes this one,
not quietly drifting.

## Context

P1 of the dense-sample initiative adds composite Copy-Paste augmentations
across four new modalities: panoptic, semantic, depth, and normals (with
instance as the existing baseline). Work across those composites happens in
separate sessions (human + agentic), so the interface-level decisions need
a pinned contract that every session can read before touching code.

Today the repository does not provide that contract:

- Stability is disclaimed globally as "pre-1.0, subject to breaking change"
  — too permissive to keep the dense-sample composites consistent with each
  other across a multi-session migration.
- `segpaste/__init__.py` re-exports 11 names but has no `__version__`, no
  `experimental` sub-namespace, no deprecation machinery, and no enumerated
  invariants that a transform must preserve.
- Randomness is split between `random` (stdlib, three call sites) and
  `torch.randint` (one call site in `processing/placement.py`). There is no
  seeding or replay contract.
- `CopyPasteConfig.blend_mode` accepts `"gaussian"` and `"poisson"` but only
  `"alpha"` is wired in `_blend_object_on_target` — a latent promise that
  will leak into the new composites unless this ADR explicitly scopes it.

This ADR fixes interface-level decisions only. Internal module layout and
algorithms are deliberately out of scope — see the final section.

---

## Part (i) — API stability and semver commitment

### The v0.9.x stability line

The surface enumerated in `segpaste/__init__.py.__all__` **plus** the new
`DenseSample` type and composite symbols introduced in P1 are semver-stable
within the v0.9.x line. "Stable" here means:

- **Additive-only.** New symbols, new fields, new optional kwargs with
  defaults preserving prior behavior — allowed.
- **No breaking changes** to signatures, return types, field names, field
  types, or invariant semantics within 0.9.x.
- **No silent behavior changes.** A release that changes the output of a
  stable transform for inputs that were previously valid is breaking.

Anything not re-exported from `segpaste.__all__` or not in
`segpaste.experimental.__all__` is private and may change in any release.

### `segpaste.__version__`

A `__version__: str` attribute is added to `segpaste/__init__.py`, populated
from the existing `hatch-vcs` dynamic version (via
`importlib.metadata.version("segpaste")`). Consumers pin against this.

### `segpaste._internal.*`

The explicit private namespace. Anything not in `segpaste.__all__` is
`_internal` regardless of its current path. P1 migrates the current
non-exported modules (`segpaste.processing.*`, parts of
`segpaste.integrations.*`, `segpaste.compile_util`, `segpaste.types.*`
beyond the exported symbols) under `_internal` or re-homes them behind
stable entry points. No intermediate state — every module is either in the
public surface or under `_internal`.

### `segpaste.experimental`

New sub-package for unstable APIs. P1 moves `CopyPasteTransform` into
`segpaste.experimental` because its API is explicitly unstable (see
`CLAUDE.md`). The top-level `segpaste.CopyPasteTransform` name is retained
as a re-export throughout 0.9.x, emitting a `DeprecationWarning` on the
first access of each process. It is removed in 0.10.0.

### `DetectionTarget` deprecation

`DetectionTarget` is superseded by `DenseSample` (see Part (iii)). It is
retained as a shim throughout 0.9.x. On construction, it emits a
`DeprecationWarning` whose message points at `DenseSample` and at this
ADR. Removal target: **0.10.0** (the next minor after the 0.9.x stability
line closes — not 1.0).

The shim is a thin subclass that forwards to `DenseSample` via the new
type's dict-level interop (see Part (iii)). No new internal code paths key
off `DetectionTarget`.

### Ambiguous `integrations` exports

`segpaste.integrations` currently exports `create_coco_dataloader` and
`labels_getter` from its own `__all__`, but neither reaches the top-level
`segpaste.__all__`. This ADR resolves the ambiguity:

- `CocoDetectionV2` — **public**, already in `segpaste.__all__`.
- `create_coco_dataloader` — **public**, promoted into `segpaste.__all__`.
- `labels_getter` — **`_internal`**, removed from any `__all__`. It is an
  internal helper for the torchvision `SanitizeBoundingBoxes` override.

### `CopyPasteConfig.blend_mode` tightening

Today `blend_mode: Literal["alpha", "gaussian", "poisson"]` but only
`"alpha"` is wired. For the 0.9.x stability line, the Literal is tightened
to `Literal["alpha"]`. Constructing `CopyPasteConfig(blend_mode="gaussian")`
raises a Pydantic validation error. `"gaussian"` and `"poisson"` remain
**reserved** names; P1 does not re-introduce them. A future ADR that wires
a new blend mode is an additive change.

---

## Part (ii) — Invariants specification

Every composite transform in P1 **MUST** preserve the invariants listed
below for its modality. Each invariant carries a `test_*` handle; Phase 1
test files map 1:1 to these handles so the spec and the test suite stay in
sync.

Notation. $U$ denotes the union of paste masks on the target image,
$\mathcal{C}_\text{st}$ the set of stuff classes, $s(p)$ the semantic label
at pixel $p$, $z(p)$ the instance id at pixel $p$ (with $0$ reserved for
stuff / background), $M_\text{eff}$ the effective paste mask after
occlusion resolution, $V$ the validity mask, $\tau$ / $\tau_\text{stuff}$
minimum-area thresholds.

### Instance

- Per-instance identity is preserved: every original instance that survives
  keeps its index and label. — `test_instance_identity_preserved`
- Target masks are subtracted by the paste-union: for every surviving
  original instance $i$, $M_i^\text{out} = M_i \setminus U$. —
  `test_instance_target_masks_subtract_paste_union`
- Bounding boxes are recomputed from masks: $\text{boxes}_i^\text{out} =
  \mathrm{bbox}(M_i^\text{out})$. —
  `test_instance_bbox_recomputed_from_mask`
- Instances with $\text{area}(M_i^\text{out}) < \tau$ are dropped. —
  `test_instance_small_area_dropped`
- No pixel is assigned to two instances of the same class. —
  `test_instance_no_same_class_overlap`

### Panoptic

- Stuff/thing consistency: $z(p) = 0 \Leftrightarrow s(p) \in
  \mathcal{C}_\text{st}$. —
  `test_panoptic_thing_stuff_consistent`
- Per-pixel bijection on thing pixels:
  $\sum_i \mathbf{1}[M_i(p) = 1] = 1$ for every $p$ with $z(p) \neq 0$. —
  `test_panoptic_pixel_bijection`
- Fresh instance ids for pasted instances:
  $\tilde z_i = \max_j z_j^b + i$, where $\max_j z_j^b$ is the max id on
  the target before paste. —
  `test_panoptic_fresh_instance_ids`
- Stuff regions survive only if remaining area
  $> \tau_\text{stuff}$; otherwise the region is merged into `ignore` or
  the neighbor stuff class per the composite's documented policy. —
  `test_panoptic_stuff_area_threshold`

### Semantic

- One class per pixel (no multi-channel semantics). —
  `test_semantic_single_class_per_pixel`
- The ignore label `255` is preserved by the composite: no paste may
  overwrite a pixel whose target label was `255`, and no composite may
  introduce `255` except via the explicit `ignore_index` of the
  `PanopticSchema` (Part (iii)). —
  `test_semantic_ignore_preserved`

### Depth

- Monotonicity on effective paste pixels:
  $d_\text{out}(p) = \min(d_\text{src}(p), d_\text{tgt}(p))$ for every
  $p \in M_\text{eff}$. —
  `test_depth_monotonicity`
- Validity join: $V_\text{out} = V_\text{src} \wedge V_\text{tgt}$
  (AND over the two boolean validity maps). —
  `test_depth_validity_join`
- Intrinsics rescale when `metric_depth=True`:
  $d_\text{src}(p) \leftarrow d_\text{src}(p) \cdot f_\text{tgt} / f_\text{src}$
  before the monotonicity step, where $f$ is the relevant focal length
  component from `CameraIntrinsics` (Part (iii)). When
  `metric_depth=False`, no rescale is performed. —
  `test_depth_metric_intrinsics_rescale`

### Normals

- Unit norm on valid pixels: $\|n(p)\|_2 = 1$ for every $p$ with
  $V(p) = \text{True}$, to within a documented tolerance $\varepsilon$
  (bit-exact equality is not required; the composite must renormalize after
  any interpolation). —
  `test_normals_unit_norm_on_valid`
- Camera-frame convention is declared as a single project-wide choice:
  **right-down-forward** (x right, y down, z into the scene). All normals
  tensors entering or leaving a composite are in this frame; transforms
  that change the frame are explicit. —
  `test_normals_camera_frame_convention`

---

## Part (iii) — Type system decisions

### `DenseSample`

`DenseSample` replaces `DetectionTarget` as the canonical per-sample
container. It is a `@dataclass(slots=True)` in `segpaste.types` with the
fields below. Validation runs in `__post_init__`, wrapped in the existing
`skip_if_compiling` decorator so it is bypassed under `torch.compile`.

| Field | Type | Shape / dtype | Required |
| --- | --- | --- | --- |
| `image` | `torchvision.tv_tensors.Image` | `[C, H, W]`, `uint8` or `float32` | always |
| `boxes` | `torchvision.tv_tensors.BoundingBoxes` | `[N, 4]`, xyxy, `float32` | always |
| `labels` | `torch.Tensor` | `[N]`, `int64` | always |
| `instance_masks` | `InstanceMask` (new `tv_tensors.Mask` subclass) | `[N, H, W]`, `bool` | when instance modality active |
| `semantic_map` | `SemanticMap` (new `tv_tensors.Mask` subclass) | `[H, W]`, `int64`, ignore label `255` | when semantic or panoptic modality active |
| `panoptic_map` | `PanopticMap` (new `tv_tensors.Mask` subclass) | `[H, W]`, `int64`, id encoding per `PanopticSchema` | when panoptic modality active |
| `depth` | `torch.Tensor` (plain, with tv-dispatch shim) | `[1, H, W]`, `float32`, meters if `metric_depth=True` | when depth modality active |
| `depth_valid` | `torch.Tensor` (plain, with tv-dispatch shim) | `[1, H, W]`, `bool` | when depth modality active |
| `normals` | `torch.Tensor` (plain, with tv-dispatch shim) | `[3, H, W]`, `float32`, unit norm on valid | when normals modality active |
| `padding_mask` | `PaddingMask \| None` | `[1, H, W]`, `bool` | optional |
| `camera_intrinsics` | `CameraIntrinsics \| None` | — | required when any composite uses `metric_depth=True` |

**Rationale for the TVTensor upgrade.** Today `image`, `boxes`,
`instance_masks` (as `masks`), and `labels` are plain `torch.Tensor` aliases
in `segpaste.types.type_aliases`. `torchvision.transforms.v2` dispatches
geometry operations (resize, crop, affine) by TVTensor subclass. Keeping
plain tensors works today because every segpaste transform short-circuits
torchvision's dispatch, but that will not generalize to the dense-sample
composites. The TVTensor upgrade is therefore a precondition for P1.

**Rationale for the plain-tensor shim on depth / normals.** Torchvision has
no first-class depth or normals types. Subclassing `Mask` would route them
through mask-kernel paths that do not respect continuous values. The shim
registers an interpolation policy per field (`bilinear` for depth /
normals, `nearest` for `depth_valid`) so `Resize` and `Crop` still
propagate through `DenseSample.to_dict() → Resize → DenseSample.from_dict()`
round-trips.

### `CameraIntrinsics`

```python
@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
```

Units are pixels (focal length and principal point in pixel coordinates, as
produced by the standard pinhole calibration). The field is required
on `DenseSample` when any composite is constructed with `metric_depth=True`;
otherwise optional. A composite that consumes intrinsics without
`metric_depth=True` set is a programming error and raises `ValueError`.

### `PanopticSchema`

```python
class PanopticSchema(Protocol):
    classes: Mapping[int, Literal["thing", "stuff"]]  # frozen
    ignore_index: int
    max_instances_per_image: int
```

Passed **explicitly** at composite construction — never inferred from the
data. `classes` is an immutable mapping (implementations typically use
`types.MappingProxyType`). `ignore_index` is the sentinel used in
`semantic_map` and `panoptic_map`. `max_instances_per_image` is an upper
bound that lets composites preallocate id ranges; exceeding it raises.

### `Modality`

```python
class Modality(enum.Enum):
    IMAGE = "image"
    INSTANCE = "instance"
    PANOPTIC = "panoptic"
    SEMANTIC = "semantic"
    DEPTH = "depth"
    NORMALS = "normals"
```

Every composite transform declares two frozenset class-vars:

```python
class PanopticCopyPaste(Transform):
    read_modalities: ClassVar[frozenset[Modality]] = frozenset(
        {Modality.IMAGE, Modality.PANOPTIC}
    )
    write_modalities: ClassVar[frozenset[Modality]] = frozenset(
        {Modality.IMAGE, Modality.PANOPTIC}
    )
```

Dispatch and validation key off these declarations. Composites that read a
modality not present on the input sample raise `ValueError` at construction
time when possible, otherwise at the first `forward` call.

### `BoundingBox` cleanup

The `BoundingBox` frozen dataclass in
`src/segpaste/types/data_structures.py` is currently orphaned — declared in
`segpaste.types.__all__` but never used internally and not in the top-level
`segpaste.__all__`. P1 removes it from `segpaste.types.__all__` and demotes
it to `segpaste._internal` if any call site surfaces; otherwise it is
deleted. It is **not** adopted by `CameraIntrinsics` (which uses plain
floats, not a rectangle abstraction).

---

## Part (iv) — Seed and replay policy

### Derived-seed hash

Each sample gets a derived seed computed as the low 64 bits of the SHA-256
digest of the canonical byte encoding of the 5-tuple
`(base_seed, epoch, rank, worker_id, sample_idx)`. The encoding is
little-endian `uint64` per field, concatenated in that order:

```python
def derive_seed(base_seed: int, epoch: int, rank: int,
                worker_id: int, sample_idx: int) -> int:
    payload = b"".join(
        int(x).to_bytes(8, "little", signed=False)
        for x in (base_seed, epoch, rank, worker_id, sample_idx)
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
```

Python's built-in `hash()` is **banned** for this purpose: it is
process-salted (`PYTHONHASHSEED`), so identical inputs across two runs
produce different ids. This ADR states it explicitly to prevent the trap
from recurring.

### Generator threading

Each sample's derived seed seeds one `torch.Generator` and one
`random.Random` instance, both threaded through the transform stack:

- All three stdlib-`random` call sites in
  `src/segpaste/augmentation/copy_paste.py`,
  `src/segpaste/augmentation/lsj.py`, and
  `src/segpaste/processing/placement.py` migrate from the module-level
  `random` functions to an injected `random.Random` instance.
- The `torch.randint` call site in
  `src/segpaste/processing/placement.py` migrates to pass
  `generator=torch.Generator` on every call.
- The two generators are constructed once per sample at the top of the
  transform pipeline (e.g. inside `CopyPasteCollator.__call__` and
  `CopyPasteTransform.forward`) and passed down; internal functions never
  re-read module-level state.

### Replay record semantic requirements

This ADR pins the **semantic requirements** for replay records; the exact
`TypedDict` shape is frozen in a follow-up ADR once loggers are in place.

- A replay record plus the base dataset plus the active `CopyPasteConfig`
  **MUST** be sufficient to reproduce any sample's augmentation
  bit-for-bit (modulo non-determinism in upstream torchvision kernels —
  where such kernels exist, P1 documents them explicitly).
- The record **MUST** carry the ADR version (`adr_version: "0001"`) so
  that future schema changes are detectable by consumers.
- The record **MUST** carry the full
  `(base_seed, epoch, rank, worker_id, sample_idx)` tuple that keys the
  derived generator.
- Records **MUST** be JSON-serializable: no tensors, no
  `torch.Generator` state objects, no opaque pickle payloads.
- Any downstream format extension during 0.9.x is **additive-only**.
  Removing a field or changing its type is breaking and blocked by the
  0.9.x stability commitment in Part (i).

---

## Gaps this ADR explicitly addresses

Called out even though they describe current state, not new work — the ADR
closes them so P1 inherits a clean slate:

- `segpaste.__version__` is missing. Added at P1 kickoff per Part (i).
- `CopyPasteConfig.blend_mode` accepts values the code ignores. Tightened
  per Part (i).
- `create_coco_dataloader` and `labels_getter` have ambiguous stability
  status. Resolved per Part (i).
- The `BoundingBox` dataclass is orphaned. Resolved per Part (iii).
- Two RNG sources today (`random` + `torch.randint`) with no seeding
  contract. Unified per Part (iv).

---

## Critical files referenced

Read-only references from this ADR (no edits in P0.A):

- `src/segpaste/__init__.py` — current `__all__` (the baseline surface).
- `src/segpaste/types/data_structures.py` — `DetectionTarget`,
  `PaddingMask`, `BoundingBox`.
- `src/segpaste/types/type_aliases.py` — the plain-tensor aliases Part
  (iii) upgrades to TVTensor subclasses.
- `src/segpaste/config.py` — `CopyPasteConfig` (the `blend_mode`
  tightening).
- `src/segpaste/augmentation/torchvision.py` — `CopyPasteTransform` (the
  symbol that moves to `experimental`).
- `src/segpaste/augmentation/copy_paste.py`,
  `src/segpaste/augmentation/lsj.py`,
  `src/segpaste/processing/placement.py` — the three `random.*` and one
  `torch.randint` call sites the seeding policy migrates.

---

## Status and supersession

- **Accepted** when this file lands on `main` with `Status: Accepted` in
  the header.
- Any later decision that contradicts Part (i)–(iv) requires a new ADR
  (`ADR-000N`) that explicitly supersedes this one, updating this file's
  `Status` to `Superseded by ADR-000N`.

## Verification

- `uv run mkdocs build --strict` passes with this ADR in the nav. Math
  blocks render via `pymdownx.arithmatex` + MathJax (both added to
  `mkdocs.yml` alongside this ADR).

## Out of scope (deliberately)

- Internal module layout for `segpaste.experimental` or the dense-sample
  composites.
- Algorithm specifics for blend modes beyond `"alpha"`.
- Migration of existing call sites (that is P1).
- Performance budget and benchmarking policy for P1.
