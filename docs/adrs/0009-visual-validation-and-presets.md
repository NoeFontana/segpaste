# ADR-0009 — Visual validation, dataset presets, and the in-repo / out-of-repo boundary

|            |                                                                                         |
| ---------- | --------------------------------------------------------------------------------------- |
| Number     | 0009                                                                                    |
| Title      | Visual-validation framework, dataset presets, license-clean repo boundary               |
| Status     | Accepted                                                                                |
| Author     | @NoeFontana                                                                             |
| Created    | 2026-04-25                                                                              |
| Updated    | 2026-04-25                                                                              |
| Tag        | `ADR-0009`                                                                              |
| Relates-to | [ADR-0001](0001-dense-sample.md) Part (i) (public surface); [ADR-0003](0003-hard-deprecation-stance.md) (hard deprecation, no RFC framing); [ADR-0008](0008-batch-copy-paste.md) §D6 (KS soft-report posture) |

## Context

`BatchCopyPaste` (ADR-0008) is the single graph-compilable entry point that
subsumes the four pre-v0.3.0 CPU wrappers. Its correctness on synthetic
fixtures is pinned by the ADR-0001 invariant matrix and the ADR-0008 §D6
KS soft-report. Neither of those answers a more practical question that
shows up the moment a user wires `BatchCopyPaste` into a real training
loop:

> *Does the augmentation produce visually plausible composites on
> dataset X with hyperparameter set Y?*

That question is dataset-specific (COCO panoptic vs. Cityscapes vs.
KITTI depth differ on every relevant axis: instance density, mask sparsity,
depth statistics, normals coverage), it cannot be answered by synthetic
Hypothesis fixtures, and it cannot be answered by a scalar metric — the
failure modes (paste off-frame, depth z-test inverted, panoptic stuff/things
collision, normals x-flip wrong sign) are all visually obvious and
metrically subtle.

The naive answer is "ship a few example renders and a `presets/` registry
that bundles dataset-specific configs." The naive answer fails on
licensing: COCO, Cityscapes, KITTI, and every other useful target dataset
ships under an attribution-required or research-only license. Distributing
imagery — even thumbnails, even contact sheets — through a PyPI wheel or
a public GitHub repo creates an attribution chain that downstream
consumers (research forks, internal productizations) inherit silently.
For a library that wants to be unconditionally redistributable, that is
the wrong default.

The framework therefore has to thread three constraints simultaneously:

1. The augmentation kernel must be visually validated against real data
   before a preset config is committed — synthetic fixtures cannot catch
   dataset-shaped failure modes.
2. The repository must remain pure code + license-clean text + derived
   numbers — no imagery, no fixtures derived from licensed data, no
   cached artifacts that would taint the redistribution chain.
3. The validation has to be reproducible and auditable: a future Claude
   session, or future-self after a six-month gap, must be able to
   re-run the same check and confirm it still holds.

This ADR pins the boundary, the file taxonomy, the sign-off ritual, and
the CI enforcement that makes the boundary mechanical rather than
aspirational.

---

## Decision

### 1. The in-repo / out-of-repo boundary

Two columns. Everything goes in exactly one.

| In repo                                              | Out of repo (local only)              |
| ---------------------------------------------------- | -------------------------------------- |
| Preset code under `src/segpaste/presets/`            | Real datasets                          |
| Synthetic invariant tests                            | Real-data contact sheets               |
| Synthetic-source KS snapshots                        | Per-sample PNG / JPEG renders          |
| `invariant_log.json` content pasted into PR body     | `local_gallery/` tree                  |
| `dataset_manifest.json` content pasted into PR body  | FiftyOne dataset directories           |
| ADRs, README, docs (text only)                       | Reviewer's own dataset copy            |
| Adapter code (license-clean, no bundled assets)      | All imagery, period                    |

The boundary is enforced by `scripts/check_no_binaries.py` (§4 below),
not by convention. Convention drifts; a CI script that exits non-zero
on a `*.png` addition does not.

### 2. Public surface

Three names land in `segpaste.__all__` at P0:

```python
from segpaste import register_preset, get_preset, list_presets, PresetConfig
```

- `register_preset(name: str, config: PresetConfig) -> None` — registers
  a preset under a unique name. Raises on duplicate names.
- `get_preset(name: str) -> PresetConfig` — returns the registered
  config. Raises `KeyError` on unknown names.
- `list_presets() -> tuple[str, ...]` — returns a sorted tuple of
  registered names. The tuple is a value, not a view into mutable state.
- `PresetConfig` — the schema below.

The registry is empty at P0. P1+ adds presets one-per-PR. Adding a
preset does not require an ADR amendment; restructuring the registry
public surface (renaming a function, changing `PresetConfig` field
semantics non-additively) does.

### 3. `PresetConfig` schema

`PresetConfig` is a frozen Pydantic v2 `BaseModel` with `extra="forbid"`,
matching the precedent set by `BatchCopyPasteConfig` (ADR-0008).
Mixed dataclass / Pydantic conventions are a papercut; the project
already chose Pydantic for its augmentation configs and presets follow.

The P0 field set is intentionally minimal. P1+ may add fields
**additively** (new optional fields with defaults that preserve prior
behavior); removing or renaming a field is breaking and follows
ADR-0001 Part (i) semantics.

```python
class PresetConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    """Stable identifier; matches the registry key."""

    description: str
    """One-paragraph human-readable rationale."""

    batch_copy_paste: BatchCopyPasteConfig
    """The augmentation hyperparameters this preset pins."""

    target_modalities: tuple[Modality, ...]
    """Which dense-sample modalities the preset expects to see."""

    sign_off: SignOff | None
    """Audit trail for the local sign-off ritual (§5). Optional at P0."""
```

`SignOff` is a sibling frozen Pydantic model carrying the run manifest
metadata (torch version, seed, sample count, ISO date of the local
ritual). It is a *value*, not a co-author credit — see §5 on why the
docstring-stamping pattern is rejected.

**Note (P6 amendment, 2026-04-26).** `BatchCopyPasteConfig` carries a
new optional `panoptic: PanopticPasteConfig | None = None` field per
the ADR-0008 amendment. Presets that target panoptic semantics (e.g.,
`coco-panoptic`) populate it; instance-only presets leave it `None`.
This is the first concrete demonstration of the §3 additive-field
contract — `BatchCopyPasteConfig` grew, `PresetConfig` did not, and
existing presets pin no panoptic-specific fields by default.

### 4. Binary-file CI guard

`scripts/check_no_binaries.py`:

- Walks the working tree (or a list of staged files passed as argv).
- Rejects any path matching
  `*.png | *.jpg | *.jpeg | *.tiff | *.tif | *.webp | *.bmp | *.gif`.
- Rejects any `*.pt | *.pth | *.ckpt | *.safetensors` file outside an
  explicit allow-list (`tests/fixtures/`, which covers the synthetic
  Hypothesis-derived fixtures under `synthetic/` and the kernel-regression
  `ks_snapshot.pt` per ADR-0008 §D6).
- Exits non-zero with a one-line-per-violation message naming the file
  and the rule.

The script is invoked from `.github/workflows/ci.yml` as a step that
runs before lint / type / test. **No `.pre-commit-config.yaml`** is
introduced — the repo currently has none, and adding a pre-commit hook
opens a "did the author install hooks?" failure mode that the CI step
already covers without it. If a future contributor wants local
pre-commit ergonomics, they install pre-commit themselves and point it
at the same script; the repo doesn't ship the config.

### 5. The local validation ritual

This project is solo + agentic (per [ADR-0003](0003-hard-deprecation-stance.md)
and the broader posture: no `CODEOWNERS`, no two-party PR review, no
engineer sign-off chain). The validation ritual reflects that. There is
no "reviewer" distinct from "author"; there is the author and a frozen
audit trail that future-self or a future Claude session can re-derive.

For each preset PR:

1. The author runs `scripts/visualize_preset.py --preset <name>` (P2)
   on a real dataset copy held locally. The script writes everything
   under `local_gallery/<preset>/`, which is gitignored.
2. The script emits two JSON artifacts:
   - `invariant_log.json` — per-sample pass/fail flags for the
     ADR-0001 invariant matrix, plus aggregates (`K_pasted`,
     paste-area fractions, KS distances against the synthetic
     baseline).
   - `dataset_manifest.json` — SHA-256 of every loaded sample, torch
     version, runner ID, seed, sample count, ISO date.
3. The author pastes the *contents* of both JSON files into the PR
   description. They contain only derived numbers and environment
   metadata — no imagery, no copyrighted strings, no filenames that
   would leak the dataset path.
4. The PR's `Status: Accepted` line in the preset's docstring is the
   acceptance criterion. Merging the PR is the acceptance.

Two patterns from earlier drafts are explicitly **not** adopted:

- **No "reviewer re-runs the script and approves" two-party flow.**
  Solo + agentic. The audit trail is the JSON content in the PR body
  plus the `SignOff` value, both reproducible from `manual_seed(0xC0FFEE)`
  on a dataset copy whose SHA-256 manifest matches.
- **No editing the preset's module docstring with a handle / SHA / date
  on merge.** That information lives in `git log` and the PR body where
  it cannot rot. Stamping it into source creates a stale-data hazard
  the moment the preset is touched again.

### 6. Reproducibility contract

Rendering and KS reduction happen on **CPU** with
`Generator().manual_seed(0xC0FFEE)`. CPU `grid_sample` is bitwise-stable
across PyTorch minor versions; CUDA `grid_sample` is not (cuDNN /
cuBLAS determinism is not guaranteed in this project — see ADR-0008
"Bitwise CPU-vs-GPU parity under matched seeds" alternative).

This means the visualizer validates **the augmentation contract** —
that a given `(PresetConfig, dataset)` pair produces the same composite
under the same seed across machines — but does **not** validate the
production GPU path byte-for-byte. A future bug where CPU and CUDA
`grid_sample` diverge at boundaries would slip past the local ritual.
The protection against that class of bug is the ADR-0008 §D7
compile-clean gate plus the ADR-0008 §D6 KS soft-report, which run on
the same CPU path. The visualizer is a complement to those gates, not
a substitute.

### 7. KS snapshot semantics, narrowed

The existing `tests/fixtures/ks_snapshot.pt` (ADR-0008 §D6) is a
*regression* gate: "did `BatchCopyPaste`'s output distribution shape
drift since the snapshot was frozen?" That semantics is preserved.

It is **not** a calibration gate against any real dataset. The synthetic
source is `dense_sample_strategy` (Hypothesis), and the snapshot answers
"did this kernel change behavior" — not "is this preset configured well
for COCO." The latter question is exclusively the local ritual's job.

This is a meaningful narrowing of what KS snapshots can claim, and it
matters because earlier framework drafts implicitly conflated the two.
Calibration is a dataset-specific property; regression is a kernel
property; mixing them produces gates that fail for the wrong reason.

### 8. Out of scope at P0

P0 lands the policy, the empty registry, and the binary guard. The
following are **deferred** to subsequent phases and are explicitly not
shipped here:

- Any actual preset (P1+). The registry is empty.
- `scripts/visualize_preset.py` (P2). Until then, the local ritual has
  no script; the boundary is enforced but the ritual is not yet
  invokable.
- Dataset adapters beyond `CocoDetectionV2` (P3+). Cityscapes / KITTI
  adapters are out of scope.
- FiftyOne integration for the local gallery (P3+).
- Any real-data CI artifact. A future ADR-0010 would have to make a
  fresh case for that; this ADR rules it out at the framework level.

## Consequences

- **Public surface delta.** `segpaste.__all__` gains `register_preset`,
  `get_preset`, `list_presets`, `PresetConfig`. `tests/test_public_surface.py`
  is updated in the same commit per ADR-0001 Part (i).
- **New `presets/` package.** `src/segpaste/presets/__init__.py` exports
  the four names; `src/segpaste/presets/_base.py` carries `PresetConfig`
  and `SignOff`. The registry is module-level state — a single private
  dict, mutated only via `register_preset`.
- **CI gains a binary-file guard step.** It runs before lint so a
  bad-shape commit fails fast. The tensor allow-list (`tests/fixtures/`)
  is explicit in the script.
- **Documentation surface unchanged at P0.** No new mkdocs pages; the
  ADR is added to the existing `nav.ADRs` list.
- **Coverage floor unchanged.** The new code is small and trivially
  covered; the 80% floor in `[tool.coverage.report]` is unaffected.
- **No new workflow_dispatch.** The only existing dispatch-only
  workflow is `bench-gpu.yml` (ADR-0008 §v); P0 adds none.

## Alternatives considered

- **Ship example renders as low-resolution thumbnails.** Discarded:
  the moment thumbnails enter the repo, every downstream redistribution
  inherits an attribution chain. The cost (no example imagery in
  public docs) is real but bounded; the cost of inheriting attribution
  is unbounded.
- **Pre-commit hook in addition to CI step.** Discarded: the repo
  currently has no pre-commit infrastructure, and adding it for one
  guard introduces a "did the author install hooks?" failure mode.
  The CI step is mechanical and unbypassable.
- **`PresetConfig` as a frozen `dataclass` instead of Pydantic.**
  Discarded: `BatchCopyPasteConfig` is Pydantic with `extra="forbid"`
  (ADR-0008), and mixing config conventions in one library is a
  papercut for users.
- **Stamp reviewer handle / SHA / date into the preset's module
  docstring on merge.** Discarded: the audit trail belongs in
  `git log` and the PR body; stamping it into source creates a
  stale-data hazard. Plus, "reviewer" doesn't exist in this project's
  posture (solo + agentic — see ADR-0003).
- **Allow real-data CI artifacts via `workflow_dispatch`.** Discarded
  at the framework level: even dispatch-only workflows that consume
  licensed data create attribution surface (cached imagery in CI
  artifact storage, secrets management for dataset access). A future
  ADR-0010 could revisit; this ADR rules it out.
- **Distribute presets in a sibling repo (`segpaste-presets`) so the
  main repo stays code-only.** Discarded: presets are tiny configs
  with strong coupling to `BatchCopyPasteConfig`'s field set. Splitting
  across repos makes additive `BatchCopyPasteConfig` migrations
  multi-repo coordination problems, which is exactly the friction the
  solo + agentic posture exists to avoid.
- **Make the KS snapshot a calibration gate against a checked-in real-data
  fixture.** Discarded: requires bundling real data, which violates §1.
  Narrowing KS to a regression-only gate (§7) keeps the in-repo claim
  honest.
