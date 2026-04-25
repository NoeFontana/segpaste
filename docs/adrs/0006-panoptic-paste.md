# ADR-0006 — `PanopticPaste`: bijection-preserving panoptic copy-paste

|            |                                                                                         |
| ---------- | --------------------------------------------------------------------------------------- |
| Number     | 0006                                                                                    |
| Title      | Panoptic-modality composite with stuff/thing gating and scatter-reduce collision policy |
| Status     | Accepted                                                                                |
| Author     | @NoeFontana                                                                             |
| Created    | 2026-04-23                                                                              |
| Updated    | 2026-04-23                                                                              |
| Tag        | `ADR-0006`                                                                              |
| Relates-to | [ADR-0001](0001-dense-sample.md) Parts (ii), (iii); [ADR-0005](0005-dense-composite.md) |

## Context

ADR-0001 Part (ii) pins four panoptic invariants — stuff/thing
consistency, per-pixel bijection on thing pixels, fresh instance ids on
paste, and stuff area threshold. The predicates are fully implemented in
`tests/invariants/panoptic.py` and parametrized as four
`InvariantRow(Modality.PANOPTIC, …, xfail=True)` entries in
`tests/test_invariant_matrix.py:125-128`, all waiting for a composite
that satisfies them.

ADR-0005 landed `DenseComposite` with image, instance, and semantic
branches. The panoptic branch is untouched: `DenseComposite.forward`
does not return `panoptic_map`, and neither `InstancePaste` nor
`ClassMix` populates it. No concrete `PanopticPaste` operator exists.

W3 closes that gap. Unlike W2 — which was anchored to a pre-rewrite
bitwise parity snapshot — there is no predecessor implementation, so
W3 is defined from first principles against the §(ii) invariants and
a hand-constructed analytical golden fixture.

## Decision

Add a `_composite_panoptic` branch to `DenseComposite` and land
`PanopticPaste` as its specialization under
`src/segpaste/_internal/panoptic_paste.py`. `PanopticPaste` reuses the
W2 `PlacementSampler` for thing placement, performs multi-instance
collision resolution via `scatter_reduce('amax')`, and delegates the
per-pixel write to `DenseComposite`. The public surface is unchanged —
`segpaste.__all__` stays frozen, `PanopticPaste` stays under
`_internal` per ADR-0005 §5.

### 1. Internal panoptic id encoding: instance-id only

`panoptic_map` stores `z(p) ∈ {0, 1, 2, …}` as the per-pixel instance
id, with `z(p) = 0` iff pixel `p` is stuff. Class information lives in
`semantic_map`; stuff/thing classification requires the schema. This
matches ADR-0001 §(ii) verbatim and every predicate in
`tests/invariants/panoptic.py` (each uses `z == 0` as the stuff check).

The COCO-panoptic-style `class_id * MAX + instance_id` encoding is
applied **only at the HuggingFace export boundary** (§6). Keeping the
internal representation as instance-ids-only means:

- the LUT used to renumber source ids via `torch.gather` is a small
  `int32[N_source]` tensor, not a dense LUT of size `num_classes * MAX`;
- the §(ii) bijection predicate is a direct `sum_i M_i == 1` check on
  `z != 0` pixels — no modular arithmetic required;
- `max_instances_per_image` is still load-bearing, as a cap that guards
  `max_prev + k` against int32 overflow across a batch.

### 2. Stuff/thing gating uses `semantic_map` + schema

A thing-mask `T ∈ {0,1}^{H×W}` is derived by looking up each pixel's
semantic class in `schema.classes` and evaluating `schema.classes[c] ==
"thing"`. Since stuff/thing discrimination requires the class label,
`PanopticPaste.transform` requires **both** SEMANTIC and PANOPTIC on
target and source and raises a `ValueError` otherwise (mirroring
`ClassMix`'s modality check at `classmix.py:59-62`).

Inside the composite, `M_eff` splits as:

- thing pixels on the target (`T_tgt`): source overwrites both class
  and instance id.
- stuff pixels on the target (`~T_tgt`): source overwrites class; the
  instance id is forced to `0`.

### 3. Conflict resolution: `scatter_reduce('amax')` with paste-order scoring

Multiple source instances can resolve to the same target pixel after
placement. The winning instance at each pixel is selected via an
`int64 [H,W]` score map built once per paste:

```
score[p] = order[i] * K + priority[c_i]  for pixel p covered by instance i
K        = 1 + max_{c} config.class_priority.get(c, 0)
```

`scatter_reduce_(reduce='amax')` selects the winner; a second `scatter`
indexed by the winning instance id writes the resolved panoptic id.
With the default `PanopticPasteConfig.class_priority = {}`, every
class gets priority 0, `K = 1`, and the score collapses to `order[i]`
— pure "later wins".

A non-empty override raises the priority of specified classes above
the paste-order term (contact-heavy scenes, e.g. "person always wins
over chair"). A dedicated property test asserts the override path is
honored on every collision.

This pattern is net-new in segpaste — there is zero existing
`scatter_reduce` / `scatter_` / `gather` usage in `src/`. `scatter_reduce_`
is CPU-deterministic but PyTorch docs flag it as non-deterministic on
CUDA under `reduce='amax'`. The parity gate therefore stays CPU-only,
matching ADR-0005 §4's CUDA statistical-equivalence convention.

### 4. Stuff area threshold lives on the config, not the schema

`PanopticPasteConfig.stuff_min_area: Mapping[int, int] = {}`. After the
composite runs, per-class pixel counts on the composed `semantic_map`
are computed; classes falling below their threshold have their pixels
reassigned to `schema.ignore_index` (on `semantic_map`) and `0` (on
`panoptic_map`).

`PanopticSchema` is already public and is the stable taxonomy.
Extending the Protocol with a `stuff_min_area` field would be a surface
change requiring an ADR-0001 amendment and a new
`_EXPECTED_PUBLIC_API` entry. Keeping the threshold on the config
avoids that and matches the composite/config separation pattern
(`CompositeConfig` vs. `CopyPasteConfig`, `CompositeConfig` vs.
`ClassMixConfig`).

### 5. Bijection enforcement: trust in release, assert in debug

`scatter_reduce('amax')` followed by a single-winner `scatter` produces
a bijective panoptic_map by construction: for each pixel, exactly one
instance id is written. Release mode trusts this and skips the
runtime check. `PanopticPasteConfig.debug_assert_bijection: bool =
False` adds the `sum_i M_i ≤ 1` assertion on thing pixels, enabled
from the fuzz test harness.

### 6. HuggingFace export: pure-torch interop, no dependency added

`src/segpaste/integrations/huggingface.py` ships two functions:

```python
def to_hf_format(sample: DenseSample, schema: PanopticSchema)
    -> dict[str, torch.Tensor]
def from_hf_format(hf: Mapping[str, torch.Tensor], schema: PanopticSchema)
    -> DenseSample
```

`to_hf_format` returns `{"mask_labels": bool[N,H,W], "class_labels":
int64[N]}` — the exact shape `Mask2FormerImageProcessor.encode_inputs`
consumes. `from_hf_format` reconstructs a `DenseSample`, assigning
fresh instance ids `1..N` and re-deriving `panoptic_map` from the
mask + label pair. The COCO-panoptic `class_id * MAX + instance_id`
encoding is applied only inside these functions, driven by
`schema.max_instances_per_image`.

Neither function imports `transformers`. Mask2Former compatibility is
structural: the `{mask_labels, class_labels}` dict shape is documented
in the `transformers` source and stable since 4.29. A round-trip test
`from_hf_format(to_hf_format(sample, schema), schema) == sample`
proves both invariant preservation and shape fidelity without
pulling in HF at test time.

### 7. Cityscapes PQ regression is out of scope

Running a SegFormer-B0 inference pass on the Cityscapes panoptic val
set requires the model weights, the dataset, and a CI story for both
— none of which the repository currently carries. The composite's
determinism is asserted by the analytical golden (§8) + the 200-seed
forward-gate snapshot; an end-to-end PQ regression is future
integration-test work.

### 8. Test strategy summary

- Analytical golden: `tests/fixtures/synthetic/panoptic_overlap.py`
  hand-constructs a `(target, source, expected)` triple for two
  overlapping things; expected argmax output is analytically derivable
  from paste order and priorities; asserted bitwise.
- Hypothesis fuzz: `dense_sample_strategy({Modality.PANOPTIC,
  Modality.SEMANTIC, Modality.INSTANCE})` with a schema-aware
  `panoptic_map_strategy` that preserves `z(p) == 0 ⟺ stuff`; zero
  bijection violations and zero ignore-leaks at `max_examples=200`.
- Forward-gate snapshot: `tests/fixtures/panoptic_baseline.pt`
  generated via `scripts/gen_panoptic_baseline.py` on CPU at W3 HEAD;
  never regenerated (same policy as ADR-0005 §4).
- Invariant-matrix flip: four `InvariantRow(Modality.PANOPTIC, …,
  xfail=True)` entries in `tests/test_invariant_matrix.py` flip to
  passing.
- W2 parity regression canary: `tests/test_dense_composite_parity.py`
  must stay bitwise — `_composite_panoptic` is only exercised when at
  least one input carries `panoptic_map`, so instance-only paths are
  untouched.

## Consequences

- `DenseComposite.forward` now returns a `DenseSample` whose
  `panoptic_map` is populated iff at least one input carried it.
  Instance-only paths are unchanged.
- `PanopticPaste`, `PanopticPasteConfig` stay under `segpaste._internal`.
  Promotion requires an ADR-0006 amendment and an
  `_EXPECTED_PUBLIC_API` entry per ADR-0001 Part (i).
- `integrations/huggingface.py` is a new top-level integration module
  under `src/segpaste/integrations/`, alongside `coco.py`. It is pure
  Python + torch; no optional extra is added to `pyproject.toml`.
- `benchmarks/_fixture.py` grows a `with_panoptic: bool = False`
  parameter, and `benchmarks/bench_panoptic_paste.py` lands alongside
  `bench_copy_paste.py` and `bench_classmix.py`.
- ADR-0005 §5's "all `_internal` symbols stay private until W3/W4
  validate the interface" moves one step closer to a follow-up
  promotion ADR: W3 exercises `DenseComposite` on a third modality
  (instance via W2 → semantic via W2 → panoptic via W3), and a second
  placement-dependent specialization (after `InstancePaste`).

## Alternatives considered

- **`class_id * MAX + instance_id` as the internal encoding.**
  Discarded: forces every panoptic invariant and fuzz predicate to go
  through a decode step (`z % MAX == 0` for stuff, `z // MAX` for
  class), breaks the direct `z == 0` predicates already in
  `tests/invariants/panoptic.py`, and turns the source-id remap from a
  small `int32[N]` `gather` into either a dense LUT of size
  `num_classes * MAX` or a decode/re-encode round-trip per paste.
  The COCO-panoptic convention is preserved for HF export (§6) where
  it matches the downstream format.
- **Add `stuff_min_area` to `PanopticSchema`.** Discarded: `PanopticSchema`
  is public. Adding a field is a surface change; adding it as optional
  via a default in a `Protocol` is not well-supported. Policy lives
  on the config.
- **First-wins or order-independent conflict resolution.** Discarded:
  first-wins is indistinguishable from "skip the paste if it would
  collide" and loses the training-signal of later, more-informative
  pastes overriding earlier ones. Order-independent (e.g. random
  tiebreak) forfeits determinism, which the parity snapshot requires.
- **Land HF interop behind a `[huggingface]` optional extra.**
  Discarded: the export function is pure torch. Adding an optional
  extra imposes an import-time surface (try/except wrapper + clear
  error message) without any reciprocal guarantee — there is no HF
  code to fail to import. Tests that actually invoke
  `Mask2FormerImageProcessor` belong in a separate downstream
  integration suite, not the repo.
- **Runtime bijection check in release.** Discarded: the scatter-reduce
  construction is mathematically bijective. A post-hoc assert on every
  `transform` burns a full-frame reduction for no added safety.
  Debug mode keeps it as a fuzz-test harness opt-in.
