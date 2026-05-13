# ADR-0013 — FiftyOne as the visualizer substrate

|            |                                                                                         |
| ---------- | --------------------------------------------------------------------------------------- |
| Number     | 0013                                                                                    |
| Title      | FiftyOne as the visualizer substrate; collapse the two-script preset ritual             |
| Status     | Accepted                                                                                |
| Author     | @NoeFontana                                                                             |
| Created    | 2026-05-12                                                                              |
| Updated    | 2026-05-12                                                                              |
| Tag        | `ADR-0013`                                                                              |
| Supersedes | [ADR-0009](0009-visual-validation-and-presets.md) §8 (FiftyOne P3+ deferral); reshapes §P2 / §P3 |
| Relates-to | [ADR-0001](0001-dense-sample.md) Part (ii) (invariant matrix); [ADR-0003](0003-hard-deprecation-stance.md) (solo + agentic posture) |

## Context

ADR-0009 §8 deferred FiftyOne integration to P3+ on the assumption that
the P2 preset visualizer would be a self-contained PNG + contact-sheet +
JSON renderer. Two things have happened since:

1. The visualizer scaffold landed at P2 (`scripts/visualize_preset.py`)
   doing exactly what §8 anticipated — per-sample drilldown PNGs,
   `contact_sheet.png`, three JSON artifacts, a `_failed/` mirror tree
   for invariant violations.
2. A parallel `scripts/fiftyone_app.py` was added (commit `9d7827f`,
   "feat(viz/eval): FiftyOne eval ritual + SanitizeInstances transform")
   that runs the same pipeline but additionally materializes a FiftyOne
   `Dataset` keyed by `sample_index`, attaches `detections` /
   `original_detections` / `stuff_segmentation` / `invariant_passed` /
   `failed_checks` / `K_pasted` / `paste_area_frac` as filterable
   fields, and optionally launches `fo.launch_app(...)` for interactive
   inspection. This work was done without amending ADR-0009 §8 — the
   docs and the code drifted.

The two scripts cover overlapping ground. The per-sample PNGs (`orig`,
`overlay`) and the `contact_sheet.png` are made redundant by FiftyOne's
native rendering: instance overlays come from `detections`, the diff
comes from `original_detections`, stuff regions come from
`stuff_segmentation`. The `_failed/` mirror is replaced by
`invariant_passed=False` filtering in the FO App. What remains useful
from `write_gallery`'s output is the augmented RGB PNG (the FO Sample's
`filepath`) and the three JSON artifacts that ADR-0009 §5 mandates for
paste-into-PR-body.

This ADR records the collapse: one script, FiftyOne as the source of
truth for audit data, JSONs as the serialization for PR review. The
in-repo / out-of-repo boundary (ADR-0009 §1) is unaffected — FiftyOne
stores under `~/.fiftyone/` and `local_gallery/` is gitignored. The
reproducibility contract (ADR-0009 §6) is unaffected — FiftyOne is
purely the output substrate; the compute path stays `DenseSample`-native
on CPU under `manual_seed(0xC0FFEE)`.

## Decision

### 1. The visualizer is a FiftyOne dataset constructor

`scripts/visualize_preset.py` is the single entry point. Its job is to
translate `(PresetConfig, sample source)` into:

- One `aug.png` per sample under `local_gallery/<preset>/samples/`.
  This is the FO Sample's `filepath`; FiftyOne renders overlays on top
  of it through the App.
- A `fo.Dataset` materialized via `_internal/viz/fiftyone_export.build_dataset()`.
  This is the **source of truth** for per-sample audit data: invariant
  outcomes, paste statistics, detections, segmentations.
- Three JSON artifacts (`invariant_log.json`, `dataset_manifest.json`,
  `run_manifest.json`) — flattened serializations of the FO fields,
  pasted into the PR description per ADR-0009 §5.

`scripts/fiftyone_app.py`, `_internal/viz/contact_sheet.py`, and
`_internal/viz/overlay.py` are deleted. `SampleOutcome.drilldown` is
removed; the writer no longer composes a contact sheet or mirrors
failed samples into `_failed/`.

### 2. Boundary unchanged

The ADR-0009 §1 in-repo / out-of-repo table stands as-is:

- `local_gallery/` is gitignored; `~/.fiftyone/` lives outside the repo.
- `scripts/check_no_binaries.py` requires no change — it already rejects
  every `*.png` outside `tests/fixtures/`.
- The `invariant_log.json` / `dataset_manifest.json` content is still
  what reviewers paste into the PR body. The JSONs now serialize the
  same data FO exposes (one source of truth) rather than a parallel
  view computed by `write_gallery`.

### 3. Reproducibility unchanged

The compute path runs on CPU under `Generator().manual_seed(0xC0FFEE)`
per ADR-0009 §6. FiftyOne is loaded *after* `BatchCopyPaste.forward`
returns; nothing inside the augmentation kernel reads from or writes to
FiftyOne. The dataset ingest path keeps using `CocoDetectionV2` →
`DenseSample` (via `_internal/viz/coco_source.py`); FiftyOne's native
importers (`COCODetectionDatasetImporter`, `KITTIDetectionDatasetImporter`,
etc.) are not used at ingest. This preserves bitwise determinism across
PyTorch minor versions.

`DatasetManifest.sha256` keys on the source-file content (computed via
`hashlib.sha256(image.tobytes())` against the loaded DenseSample), not
on FiftyOne's nondeterministic sample IDs. This stays the audit-trail
anchor.

### 4. Persistence is opt-in

`fo.Dataset(persistent=False)` is the default for one-shot audit runs —
the visualizer should not accumulate MongoDB cruft under `~/.fiftyone/`
on every run. The CLI exposes `--persist NAME` which sets
`dataset.persistent=True` and uses `NAME` as the dataset name (mutually
exclusive with `--dataset-name`), supporting the reviewer who wants to
compare two preset configs side-by-side in the App across sessions.

`--launch` is also opt-in (default off). Running the script for the
audit ritual never blocks on a browser session; reviewers paste the
JSONs into the PR body without needing the FO App to be open. The App
is for deeper inspection.

### 5. Extras: `[visualize]` replaces `[viewer]`

The `[dependency-groups].viewer` group is renamed to `visualize`
(`fiftyone>=1.14.2` unchanged). The vestigial `fiftyone>=1.8.0` pin
inside `[dependency-groups].coco` is removed — `CocoDetectionV2` imports
`faster_coco_eval`, not `fiftyone`, so the pin was inherited from an
earlier design.

This stays in `[dependency-groups]` (PEP 735, uv-only). Promoting to
`[project.optional-dependencies]` so `pip install segpaste[visualize]`
works end-to-end for non-uv consumers is deferred — no current consumer
needs it, and adding it carries a small wheel-metadata cost.

`require_fiftyone()` in `_internal/imports.py` updates its install hint
to `uv sync --group visualize`.

### 6. Invariant coverage gap

The post-hoc viz layer dispatches 6 of the 15 ADR-0001 §Part-(ii)
invariants via `_internal/viz/invariant_runner.run_invariants(before, after)`:

- `semantic.single_class_per_pixel`, `semantic.ignore_preserved`
- `instance.no_same_class_overlap`, `instance.identity_preserved`
- `panoptic.pixel_bijection`
- `normals.unit_norm_on_valid`, `normals.camera_frame_convention`

The remaining 9 require additional context that `run_invariants`'s
`(before, after)` signature cannot supply: `paste_union` (from
`TileCompositor`), `PanopticSchema` (from `PanopticPasteConfig`), `tau`
thresholds (from `BatchCopyPasteConfig.{small_area_min, tau_stuff_frac}`),
and camera intrinsics for depth metric rescaling. Plumbing these
through `BatchCopyPaste.forward`'s return surface is a separate piece
of work and follows its own ADR.

For A4, the FO Sample's `failed_checks` field lists only the names of
the 6 reachable checks that failed; passing those 6 is necessary but
not sufficient for full ADR-0001 §Part-(ii) compliance, and the PR
review ritual reflects that.

**Closed by [ADR-0014](0014-batchauditpacket-forward-return-sidecar.md)**:
the new `BatchCopyPaste.forward_with_audit` sibling method returns a
`BatchAuditPacket` carrying the post-z-test paste union, warped source
depth fields, source intrinsics, panoptic schema, and fractional area
thresholds. `run_invariants` is widened to `(before, after, audit=None)`
and the viz pipeline at `_internal/viz/pipeline.py:72` now calls
`forward_with_audit`. Audited coverage is **15 of 16** §Part-(ii)
invariants (`depth.metric_intrinsics_rescale` is carved out of the
audit-path dispatch and pinned by `tests/test_invariants_internal.py`
at the wrapper level — see ADR-0014 §4). The `count of 6 of 15` in the
heading above is a pre-ADR-0014 baseline measurement; the canonical
count in ADR-0001 §Part (ii) is 16 (5 instance + 4 panoptic + 2 semantic
+ 3 depth + 2 normals).

## Consequences

- **Public surface unchanged.** `segpaste.__all__` is untouched;
  `tests/test_public_surface.py` does not need amending.
- **Two scripts collapse to one.** `scripts/fiftyone_app.py` is deleted;
  `scripts/visualize_preset.py` is rewritten to be the merged entry
  point with the new defaults (`--source {synthetic,coco}`, `--launch`
  opt-in, `--persist NAME` opt-in).
- **Three internal modules retired.** `_internal/viz/contact_sheet.py`,
  `_internal/viz/overlay.py`, and `SampleOutcome.drilldown` are
  removed. `write_gallery` now writes only `aug.png` (one per sample)
  plus the three JSONs.
- **Tests update.** `tests/test_fiftyone_integration.py` drops the
  `original_filepath` / `overlay_filepath` `is_file()` assertions.
  `tests/test_visualize_preset_smoke.py` is rewritten to assert the
  new artifact layout (only `aug.png` per sample, no contact sheet, no
  `_failed/`), and gates on `pytest.importorskip("fiftyone")` because
  the script always builds the FO Dataset.
- **`scripts/build_eval_subset.py`** updates its README-template
  references to point at `scripts/visualize_preset.py --launch`
  (instead of `scripts/fiftyone_app.py`) and at `--group visualize`
  (instead of `--group viewer`).
- **CLAUDE.md correction.** The `fiftyone is an optional extra
  (uv sync --extra coco) used by create_coco_dataloader` line is
  updated to reflect reality — FO is now the `--group visualize`
  visualizer-only dependency, and `create_coco_dataloader` does not
  use it.
- **No CI guard change.** `scripts/check_no_binaries.py` already
  rejects PNG / tensor additions outside the allow-list; the gallery
  PNGs land under the gitignored `local_gallery/` and never enter
  staged changes.
- **Coverage floor unchanged.** Net LOC delta is negative (~200 LOC
  deleted between contact_sheet.py / overlay.py / drilldown plumbing /
  `_failed/` mirror); the 80% floor in `[tool.coverage.report]` holds.

## Alternatives considered

- **Keep both scripts and have FiftyOne layer on top.** Discarded:
  duplicate ingest, duplicate exit-code semantics, duplicate
  documentation surface. The two-script state was a transient
  consequence of how the FO ritual landed; collapsing it is the
  intended end state per the original §P2 design (just now built on
  FO).
- **Use FiftyOne's native importers for dataset ingest.** Discarded:
  couples the compute path to FO (the converter from
  `fo.Sample → DenseSample` becomes part of the augmentation contract,
  which ADR-0009 §6 explicitly rules out). Multi-format ingest (KITTI,
  VOC, …) for free is real value, but it can be added separately
  without dragging FO into the seed-deterministic compute layer.
- **Promote `[visualize]` to `[project.optional-dependencies]` so
  `pip install segpaste[visualize]` works.** Deferred: no current
  consumer needs it (the project is uv-first), and the cost of doing
  it later is small.
- **Default `dataset.persistent=True`.** Discarded: one-shot audit
  runs would accumulate MongoDB cruft under `~/.fiftyone/`; reviewers
  who want persistence can opt in with `--persist NAME`.
- **Default `--launch=True` (open the App on every run).** Discarded:
  the audit ritual is a script invocation in CI-like local conditions;
  blocking on a browser session by default is the wrong shape.
- **Widen `run_invariants` to dispatch all 15 ADR-0001 §Part-(ii)
  invariants in this ADR.** Deferred: requires returning
  `paste_union` / `PanopticSchema` / `tau` / intrinsics from
  `BatchCopyPaste.forward` (or sidecar plumbing), which is a separate
  design decision. Documenting the gap is honest; widening it in the
  same PR conflates two unrelated concerns.
- **Stamp the FO dataset name with a content hash for deterministic
  cross-run identity.** Discarded: the dataset is ephemeral by
  default; `--persist NAME` lets the reviewer choose a stable handle
  when they actually want one. Stamping by content hash would also
  surprise reviewers running back-to-back invocations of the same
  config (each would clobber the previous run unless `overwrite=False`,
  which `build_dataset` does not pass today).
