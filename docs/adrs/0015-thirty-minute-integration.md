# ADR-0015 — 30-minute integration story

|            |                                                                                                          |
| ---------- | -------------------------------------------------------------------------------------------------------- |
| Number     | 0015                                                                                                     |
| Title      | Preset exposure, framework adapters, and end-to-end quickstart docs                                      |
| Status     | Accepted                                                                                                 |
| Author     | @NoeFontana                                                                                              |
| Created    | 2026-05-24                                                                                               |
| Updated    | 2026-05-24                                                                                               |
| Tag        | `ADR-0015`                                                                                               |
| Amends     | [ADR-0001](0001-dense-sample.md) Part (i) (public surface adds 6 names)                                  |
| Relates-to | [ADR-0003](0003-hard-deprecation-stance.md) (hard delete); [ADR-0009](0009-visual-validation-and-presets.md) §3 (presets registry); [ADR-0008](0008-batch-copy-paste.md) (compile-clean kernel) |

## Context

A new user `pip install segpaste`-ing today has no path to a working
training-loop integration shorter than reading the source. The README's
Usage section (lines 21-54) demonstrates `BatchCopyPaste()` with defaults
but no preset, no model, and no training step. `docs/index.md` is a one-
line include of the README; `docs/api.md` is a single `::: segpaste`
mkdocstrings directive. There is no migration page from the prior
copy-paste implementations (`torchvision.transforms.v2.SimpleCopyPaste`,
`mmdet.datasets.transforms.CopyPaste`); no Hugging Face Trainer recipe;
no Lightning adapter.

Two pieces of the puzzle already exist:

1. The `coco-instance` and `coco-panoptic` presets are registered at
   `src/segpaste/presets/coco_instance.py` and
   `src/segpaste/presets/coco_panoptic.py` (ADR-0009 §3). Their public
   API — `get_preset` / `list_presets` / `register_preset` /
   `PresetConfig` — is already in `segpaste.__all__`.
2. The Hugging Face per-sample converters `to_hf_format` /
   `from_hf_format` exist at `src/segpaste/integrations/huggingface.py`
   (ADR-0006 §6) but are not yet in the public surface and do not have a
   batch-level pairing.

The gap is **discoverability, batch-level glue, and documentation**. This
ADR records a bundled fix for all three, with a 30-minute end-to-end
ergonomics budget as the acceptance gate.

The set of frameworks targeted is intentionally narrow: torchvision
(already a hard dependency), Hugging Face Transformers (structural-only
compatibility through `Mask2FormerImageProcessor`), and PyTorch
Lightning. Detectron2 and mmdet are explicitly out of scope — both ship
their own data pipelines that compete rather than compose with
segpaste's `PaddedBatchedDenseSample` flow; either would require a deep
fork to integrate cleanly. The `[lightning]` extra is the only new
optional dependency the wheel needs to carry.

## Decision

### 1. Framework adapters

Each adapter is a single module under `src/segpaste/integrations/` with
a 200-line ceiling. Lazy import discipline is preserved — the adapter
must not import its target framework at module top.

| Framework    | File                                                | Public verbs                                                                                            |
| ------------ | --------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| torchvision  | `integrations/torchvision.py` (NEW)                 | `make_segpaste_collate_fn(max_instances=32)`                                                            |
| Hugging Face | `integrations/huggingface.py` (extend in place)     | `to_hf_format`, `from_hf_format`, `to_hf_batch`, `make_hf_collate_fn(preset_name, max_instances=32)`    |
| Lightning    | `integrations/lightning.py` (NEW)                   | `make_segpaste_datamodule(preset_name, train_dataset, val_dataset=None, batch_size=8, max_instances=32)` |

Conventions:

- **No wrapper class for `BatchCopyPaste`.** Users construct
  `BatchCopyPaste(get_preset(name).batch_copy_paste)` directly. Wrapping
  it adds public surface without value.
- **HF batch shape is list-of-tensors.** `to_hf_batch` emits
  `{mask_labels: list[Tensor], class_labels: list[Tensor], pixel_values: Tensor[B, C, H, W]}`,
  matching the output of `Mask2FormerImageProcessor.encode_inputs` and
  the training-time `forward` signature of
  `Mask2FormerForUniversalSegmentation`.
- **Lightning adapter is a factory.** The `LightningDataModule` subclass
  is constructed inside `make_segpaste_datamodule` after a lazy
  `_require_lightning()` call. The class body never runs at module
  import. Consistent with the `make_large_scale_jittering` style
  (`src/segpaste/augmentation/lsj.py:148`).
- **Augmentation runs in `on_after_batch_transfer`** on the Lightning
  `DataModule`, on the GPU side of the device transfer — matches
  `BatchCopyPaste`'s compile-clean contract (ADR-0008 §D7).
- **Lazy-import gate.** A new `_require_lightning() -> ModuleType` is
  added to `src/segpaste/_internal/imports.py`, mirroring
  `require_fiftyone()` and `require_huggingface_hub()`. Raises
  `ImportError("Install with `pip install \"segpaste[lightning]\"`.")` on miss.

### 2. Optional dependencies

- **ADD** `[lightning]`: pinned `lightning>=2.6`. This is the canonical
  PyPI distribution name since the 2.0 rename of `pytorch-lightning`;
  it provides `lightning.pytorch.LightningDataModule`, which is the
  import path the adapter uses. `2.6` is the latest stable as of the
  ADR-acceptance date (2026-05-24).
- **SKIP** `[huggingface]`: the HF adapter performs no runtime import
  of `transformers`. Listing the extra would mislead users into
  thinking it is required. The getting-started page documents the
  `transformers` install as a separate user concern.
- **SKIP** `[torchvision]`: already a hard dependency in
  `[project.dependencies]` (`pyproject.toml:26`).

### 3. Public surface (amends ADR-0001 Part (i))

Six names are added to `segpaste.__all__` in
`src/segpaste/__init__.py` and to the pinned tuple
`_EXPECTED_PUBLIC_API` in `tests/test_public_surface.py`:

- `from_hf_format` (already-defined, exposed now)
- `make_hf_collate_fn` (new)
- `make_segpaste_collate_fn` (new)
- `make_segpaste_datamodule` (new)
- `to_hf_batch` (new)
- `to_hf_format` (already-defined, exposed now)

All six are functions; no classes are added to the top-level surface.

### 4. Documentation

The `mkdocs` nav (`mkdocs.yml:60-78`) is restructured from a flat list
into five top-level entries: Home, Getting started, Migration guide,
Design principles, Reference (Public API / Presets / Integrations), and
Decision records (overview + ADR list). New pages:

- `docs/getting-started.md` — ≤ 50-line runnable example covering
  install → COCO panoptic dataset → preset → `BatchCopyPaste` → Hugging
  Face Mask2Former → one `loss.backward()` step. The reader supplies the
  COCO root path; no bundled fixtures.
- `docs/migration.md` — two H2 sections (torchvision SimpleCopyPaste,
  mmdet.CopyPaste). Each uses `pymdownx.tabbed` before/after blocks and
  a `!!! warning "Semantic difference"` admonition flagging GPU-vs-CPU,
  batched-vs-per-sample, and panoptic support.
- `docs/design-principles.md` — five short sections (DenseSample as
  canonical container; hard delete posture; GPU-resident + compile-clean
  with the empty allow-list contract; the invariant matrix; practical
  implications). Pointers to ADR-0001/0003/0008/0014, not regurgitation.
- `docs/reference/api.md` — replaces `docs/api.md`. Explicit `:::`
  directives per `__all__` entry, grouped under H2s (Augmentation, Types,
  Presets, Integrations). Robust against `__all__` drift because every
  reference is named.
- `docs/reference/presets.md` — prose for the two registered presets
  plus mkdocstrings reference to their `PresetConfig`. Surface for what
  `list_presets()` returns.
- `docs/reference/integrations.md` — tabbed by framework (COCO, Hugging
  Face, torchvision, Lightning). Pure `:::` references only; no
  in-progress stubs.
- `docs/adrs/index.md` — one-screen subsystem map.

`docs/index.md` is rewritten as a 5-link landing card; the
`--8<-- "README.md"` include is dropped (the docs site stops being a
mirror of the GitHub README). The README's own Usage section is trimmed
to a 2-line pointer to the docs site — duplication forbidden by
ADR-0003.

## Consequences

- **New optional dependency.** `[lightning]` is added; the base wheel
  remains slim.
- **Surface grows by six names.** Every addition is a function with a
  clear single responsibility. The `test_public_surface.py` pin is the
  forcing function preventing accidental further growth (ADR-0001 Part
  (i)).
- **Lazy-import discipline extended.** `_require_lightning` joins the
  existing import gateways in `src/segpaste/_internal/imports.py`. The
  HF adapter remains gateway-free (it imports nothing from `transformers`).
- **Compile-clean contract unchanged.** Adapters wrap, never replace,
  `BatchCopyPaste.forward`. The empty allow-list at
  `scripts/compile_allowlist.txt` is unaffected (ADR-0008 §D7).
- **README becomes a landing card.** The Usage snippet is removed; the
  docs site is now the single source for runnable examples. ADR-0003's
  "no duplication" stance is honored.
- **Docs CI gate strengthens.** `mkdocs build --strict` continues to run
  on every push, and now polices a substantially larger surface. The
  `tests/test_public_surface.py` gate and the `mkdocs --strict` gate
  together ensure that any future addition to `__all__` either appears
  in `docs/reference/api.md` or fails CI.

## Out of scope

- **detectron2 and mmdet adapters.** Both ship competing data pipelines;
  integrating would require a deep fork rather than a thin adapter. A
  future ADR may revisit if user demand materializes.
- **Native HF `Trainer` subclass.** `make_hf_collate_fn` plus a
  user-supplied `Trainer` is sufficient. A subclass would add framework
  surface coupling without a 200-line ceiling justification.
- **`SegPasteAugmenter` convenience wrapper.** Construction is already
  one line; the wrapper would obscure the preset → kernel relationship.
- **Bundled COCO fixtures.** ADR-0009 §1's in-repo / out-of-repo
  boundary forbids dataset binaries in the repo; the getting-started
  example accepts the COCO root as a user-supplied path.
