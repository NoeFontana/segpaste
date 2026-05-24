# Design principles

Five conventions shape what segpaste does and does not do. Each links
back to the ADR that records the decision in full.

## `DenseSample` is the canonical container

Every public verb in segpaste operates on one of three containers:
`DenseSample` (per-sample, ADR-0001), `BatchedDenseSample` (ragged
batch, ADR-0004), or `PaddedBatchedDenseSample` (K-padded batch,
ADR-0008). Modalities — `image`, `instance_masks`, `semantic_map`,
`panoptic_map`, `depth`, `normals`, `padding_mask`, `camera_intrinsics`
— are explicit fields on the dataclass. There is no `dict`-of-tensors
escape hatch.

A user who wants to wire a new modality writes a `DenseSample` instance;
a kernel that consumes one or more modalities reads explicit fields.
Shape and consistency invariants (ADR-0001 Part (ii)) are checked in
`__post_init__` outside compiled regions and bypassed inside via the
`skip_if_compiling` decorator (`compile_util.py`).

→ See [ADR-0001](adrs/0001-dense-sample.md) and
[ADR-0004](adrs/0004-batched-dense-sample.md).

## Hard delete, not deprecate

segpaste is pre-1.0 (ADR-0001 Part (i)) and follows a strict hard-delete
posture (ADR-0003). When an API changes, the old one disappears in the
same commit. There is no `_deprecated/` namespace, no runtime warning
shim, no `__getattr__` re-export. The pinned public surface lives in
`segpaste.__all__` and is policed by `tests/test_public_surface.py`;
amending it requires amending the test's `_EXPECTED_PUBLIC_API` tuple,
which in turn requires an ADR amendment.

The consequence: every example in this documentation runs against the
current release. No stale snippets.

→ See [ADR-0003](adrs/0003-hard-deprecation-stance.md).

## GPU-resident, compile-clean

`BatchCopyPaste.forward` is the single GPU-resident kernel that replaces
the pre-v0.3.0 CPU collator and the four modality-specific wrappers
(`InstancePaste`, `PanopticPaste`, `DepthAwarePaste`, `ClassMix`). It
must trace cleanly under `torch.compile(fullgraph=True)` — the
`tests/test_compile_clean.py` gate runs `torch._dynamo.explain` against
the empty allow-list at `scripts/compile_allowlist.txt` and fails on
any new graph break.

Adding a graph break requires both an ADR amendment (ADR-0008 §D7) and
a justified addition to the allow-list. The allow-list is empty today
and is intended to stay empty.

→ See [ADR-0008](adrs/0008-batch-copy-paste.md).

## The invariant matrix is explicit

ADR-0001 Part (ii) defines a matrix of per-modality invariants
(panoptic pixel bijection, semantic same-class overlap, depth
monotonicity, normals unit-length, etc.). The invariants live in
`src/segpaste/_internal/invariants/` as paired
`check_* -> InvariantReport` (non-raising) and `assert_*` (raising)
predicates. Tests dispatch them via Hypothesis strategies; the visual
audit dispatches them via the `BatchAuditPacket` sidecar (ADR-0014).
Today's ceiling is 15 of 16 invariants dispatchable from the audit
path; the carve-out is `depth.metric_intrinsics_rescale` (ADR-0014 §4).

A reader does not need to memorize the matrix — but knowing it exists
clarifies why `BatchCopyPaste` returns a `BatchAuditPacket` alongside
the augmented batch on its audited path: every modality's invariants
are checked from the same data.

→ See [ADR-0001](adrs/0001-dense-sample.md) Part (ii) and
[ADR-0014](adrs/0014-batchauditpacket-forward-return-sidecar.md).

## What this means in practice

- **All augmentation runs post-collate.** `BatchCopyPaste` is an
  `nn.Module`; it lives in the training loop, not in the dataset.
- **Presets are the public knob set.** Construct
  `BatchCopyPaste(get_preset(name).batch_copy_paste)`; do not pass
  hand-rolled kwargs.
- **Configs are frozen.** `BatchCopyPasteConfig`, `PresetConfig`, and
  every nested model are Pydantic v2 with `frozen=True` and
  `extra="forbid"`. Override via `model_copy(update={...})`.
- **No compat shims.** Renaming, moving, or removing a public name is
  a single-commit operation governed by the ADR + surface-pin pair.
- **Optional extras stay slim.** Today: `[lightning]` for the
  PyTorch Lightning DataModule factory, `[bank-*]` for the
  InstanceBank backends, `[visualize]` for the FiftyOne visualizer.
  Hard dependencies are limited to torch, torchvision,
  faster-coco-eval, and pydantic.
