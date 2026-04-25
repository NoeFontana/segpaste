# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

PyTorch reimplementation of "Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation" ([arXiv:2012.07177](https://arxiv.org/abs/2012.07177)). Integrates with the `torchvision.transforms.v2` ecosystem.

The package is pre-1.0: the public API exposed in `segpaste/__init__.py` is subject to breaking changes.

## Development commands

The project uses [`uv`](https://docs.astral.sh/uv/) for environment/dependency management. CI runs on Python 3.11, 3.12, and 3.13.

```bash
uv sync                          # install dev deps into .venv
uv run ruff format --check .     # format check (CI gate)
uv run ruff check .              # lint (CI gate)
uv run pyright                   # strict type check (CI gate)
uv run pytest                    # full test suite (runs with coverage per pyproject.toml)
uv run pytest tests/test_batch_copy_paste_shape.py::test_name   # single test
uv run python scripts/compile_explain.py --allowlist scripts/compile_allowlist.txt
```

`make format` auto-fixes lint + formatting across `src` and `tests`. `make test-cov` runs pytest with HTML coverage.

Pyright runs in `strict` mode with `include = ["src", "tests", "benchmarks"]`. The four `reportUnknown*` / `reportMissingTypeStubs` rules are downgraded from `error` to `warning` (per ADR-0001 Part (i)'s audit) because every first-party hit comes from a line interacting with `torchvision` / `fiftyone` / `faster_coco_eval`, none of which ship complete stubs. CI exits 0 on warnings; only errors gate merges. Coverage floor is pinned at 80% in `[tool.coverage.report].fail_under` — raise only when the measured floor moves up.

## Architecture

### Sample containers (`types/`)

`DenseSample` is the per-image canonical container: an `image`, optional `instance_masks`/`boxes`/`labels`/`instance_ids`, optional `semantic_map`, `panoptic_map`, `depth` + `depth_valid` + `metric_depth` flag, `normals`, `camera_intrinsics`, and an optional `padding_mask`. `BatchedDenseSample` stacks these with ragged per-sample instance counts. `PaddedBatchedDenseSample` is the `[B, K, ...]` sibling produced by `BatchedDenseSample.to_padded(max_instances)` — every field is leading-batch with an `instance_valid [B, K]` gate so the forward pass traces under `torch.compile(fullgraph=True)`. Shape/consistency validators in `__post_init__` are bypassed inside compiled regions via the `skip_if_compiling` decorator in `compile_util.py`.

`PaddingMask` subclasses `tv_tensors.Mask` but is semantically *not* tied to an object instance — it marks padded pixels of an image. The reimplemented `SanitizeBoundingBoxes` in `augmentation/lsj.py` forwards `PaddingMask` instances unchanged (the stock torchvision version would try to filter them alongside per-object masks).

### Public entry point

**`BatchCopyPaste`** (`augmentation/batch_copy_paste.py`) — `nn.Module` consuming a `PaddedBatchedDenseSample` and returning one. Replaces every CPU wrapper from the pre-v0.3.0 stack (instance, panoptic, depth-aware, classmix) under a single graph-compilable forward. Configuration is `BatchCopyPasteConfig`, a frozen Pydantic model with `extra="forbid"`. The compile-clean invariant is pinned by `tests/test_compile_clean.py` against the empty allow-list at `scripts/compile_allowlist.txt` (additions require an ADR amendment per ADR-0008 §D7).

### GPU pipeline (`_internal/gpu/`)

`BatchCopyPaste.forward` runs three stages, all leading-batch tensor:

1. `BatchedPlacementSampler` (`batched_placement.py`) draws one affine per target slot — `(source_idx, scale, translate, hflip)` — under a diagonal-masked `torch.multinomial` so every target picks a source `!= i`. Returns a `BatchedPlacement` plus a `paste_valid [B, K]` gate.
2. `AffinePropagator` (`affine_propagate.py`) applies the sampled affine to every channel group of the selected source via `grid_sample` — bilinear for continuous modalities (image, depth, normals), nearest for label modalities (instance/semantic/panoptic, depth_valid). Hflip applies the normals x-sign-flip per ADR-0007 §7.
3. `TileCompositor` (`tile_composite.py`) iterates over fixed-size tiles and runs `DenseComposite._effective_mask`'s z-test (ADR-0005 §3) per-tile so activation memory scales with `tile_size²`.

The pixelwise where-composite primitive lives in `_internal/composite.py::DenseComposite` and is consumed both by the tile compositor and by direct unit tests (`tests/test_dense_composite.py`).

### LSJ helpers (`augmentation/lsj.py`)

`make_large_scale_jittering` composes `RandomResize` + `FixedSizeCrop` and is the canonical preprocessing pipeline used in the paper. These are `torchvision.transforms.v2.Transform` subclasses; follow the same `make_params` / `transform` split when extending them.

### COCO integration (`integrations/coco.py`)

`CocoDetectionV2` wraps `faster_coco_eval`'s `COCO` and yields `(image, target_dict)` where `target_dict` is already shaped for transforms v2 (`tv_tensors.BoundingBoxes` in XYXY, `tv_tensors.Mask`, optional `PaddingMask` of zeros). `segpaste/__init__.py` calls `faster_coco_eval.init_as_pycocotools()` at import time to shim the pycocotools API. `fiftyone` is an optional extra (`uv sync --extra coco`) used by `create_coco_dataloader`.

## Public surface

`segpaste.__all__` is pinned by `tests/test_public_surface.py::test_all_matches_pinned_surface`. Adding or removing a top-level public name requires amending both the `__all__` list and the `_EXPECTED_PUBLIC_API` tuple in the test, which in turn requires an ADR amendment (ADR-0001 Part (i) / ADR-0003). `segpaste._internal` is the reserved namespace for private modules P1 re-homes during the dense-sample migration.

## Testing

- `tests/strategies/` hosts the Hypothesis strategies for ADR-0001 per-modality invariants. The invariants themselves live in `src/segpaste/_internal/invariants/` as paired `check_* -> InvariantReport` (non-raising) and `assert_*` (raising) predicates, shared by the test suite and the visualizer (P5+).
- `tests/shared.py` provides canonical LSJ / resize transform pipelines — reuse these rather than rebuilding similar pipelines in new tests.
- `tests/test_public_surface.py` enforces the ADR-pinned top-level API; see §Public surface.
- `tests/test_compile_clean.py` runs `torch._dynamo.explain` on `BatchCopyPaste.forward` against a fixture batch and fails on any graph break outside `scripts/compile_allowlist.txt`.
- `tests/test_batch_copy_paste_ks.py` is a soft-report KS gate against `tests/fixtures/ks_snapshot.pt` (frozen pre-deletion CPU-wrapper outputs); writes `ks_report/` as a CI artifact, no assertion at v0.3.0 (ADR-0008 §D6 burn-in).
- `pytest` is configured with `--strict-markers --strict-config` and coverage always on; don't add ad-hoc markers without registering them.
