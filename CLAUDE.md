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
uv run pytest tests/test_copy_paste.py::test_name   # single test
```

`make format` auto-fixes lint + formatting across `src`, `tests`, and `examples`. `make test-cov` runs pytest with HTML coverage.

Pyright runs in `strict` mode with `include = ["src", "tests", "examples", "benchmarks"]`. The four `reportUnknown*` / `reportMissingTypeStubs` rules are downgraded from `error` to `warning` (per ADR-0001 Part (i)'s audit) because every first-party hit comes from a line interacting with `torchvision` / `fiftyone` / `faster_coco_eval`, none of which ship complete stubs. CI exits 0 on warnings; only errors gate merges. Coverage floor is pinned at 80% in `[tool.coverage.report].fail_under` — raise only when the measured floor moves up.

## Architecture

### Data flow around `DetectionTarget` (0.9.x internal-only)

`segpaste.types.DetectionTarget` is the legacy instance-only container used by every augmentation step. It bundles `image [C,H,W]`, `boxes [N,4]` in xyxy, `labels [N]`, `masks [N,H,W]`, and an optional `padding_mask [1,H,W]`. It has `from_dict` / `to_dict` helpers so transforms can accept torchvision's dictionary-style targets and round-trip through copy-paste. Shape/consistency validation happens in `__post_init__` but is bypassed under `torch.compile` via the `skip_if_compiling` decorator in `compile_util.py`.

As of 0.9.0 `DetectionTarget` is removed from `segpaste.__all__` and `segpaste.types.__all__` — it is internal-only. P1's W1 workstream migrates every transform to `DenseSample` and deletes the class in 0.9.1. Conversion uses the bidirectional static methods `DenseSample.from_detection_target` / `DenseSample.to_detection_target`. See [ADR-0003](docs/adrs/0003-hard-deprecation-stance.md).

`PaddingMask` subclasses `tv_tensors.Mask` but is semantically *not* tied to an object instance — it marks padded pixels of an image. The reimplemented `SanitizeBoundingBoxes` in `augmentation/lsj.py` forwards `PaddingMask` instances unchanged (the stock torchvision version would try to filter them alongside per-object masks).

### Public entry point

**`CopyPasteCollator`** — a drop-in `collate_fn` for `DataLoader` that treats every object in the batch as a potential source for every other image in the batch. Requires `batch_size > 1`. Lives in `augmentation/torchvision.py`. Delegates to `CopyPasteAugmentation.transform(target, source_objects)`.

### Copy-paste pipeline (`augmentation/copy_paste.py`)

`CopyPasteAugmentation` orchestrates a single image's augmentation:

1. Roll against `paste_probability`; sample `random.randint(min_paste_objects, max_paste_objects)` source objects.
2. Validate each source with `_is_valid_object` (min edge, min area).
3. For each object, call `_try_place_single_object` → `processing.placement.create_object_placer` → `ObjectPlacer.find_valid_placement`. Placement respects existing pasted boxes (IoU-based collision) and the target's padding mask.
4. Alpha-blend the object in `_blend_object_on_target`, then stack a per-object binary mask into target coordinates.
5. `_update_and_filter_occluded_objects` subtracts new masks from originals, recomputes boxes via `torchvision.ops.masks_to_boxes`, and drops any original whose remaining area ratio exceeds `occluded_area_threshold`.

Configuration is `segpaste.config.CopyPasteConfig`, a frozen Pydantic model with `extra="forbid"` — unknown fields raise.

### Placement strategies (`processing/placement.py`)

Uses a Protocol + composition pattern: `PlacementGenerator` produces candidates, a list of `PlacementValidator` filters them. `create_object_placer` picks `PaddingAwarePlacementGenerator` when a padding mask is supplied, otherwise `RandomPlacementGenerator`, and always wires `BoundsValidator` + `OverlapValidator`. When adding new placement behaviors, implement the Protocols rather than editing `ObjectPlacer`.

### LSJ helpers (`augmentation/lsj.py`)

`make_large_scale_jittering` composes `RandomResize` + `FixedSizeCrop` and is the canonical preprocessing pipeline used in the paper. These are `torchvision.transforms.v2.Transform` subclasses; follow the same `make_params` / `transform` split when extending them.

### COCO integration (`integrations/coco.py`)

`CocoDetectionV2` wraps `faster_coco_eval`'s `COCO` and yields `(image, target_dict)` where `target_dict` is already shaped for transforms v2 (`tv_tensors.BoundingBoxes` in XYXY, `tv_tensors.Mask`, optional `PaddingMask` of zeros). `segpaste/__init__.py` calls `faster_coco_eval.init_as_pycocotools()` at import time to shim the pycocotools API. `fiftyone` is an optional extra (`uv sync --extra coco`) used only by the COCO example.

## Public surface

`segpaste.__all__` is pinned by `tests/test_public_surface.py::test_all_matches_pinned_surface`. Adding or removing a top-level public name requires amending both the `__all__` list and the `_EXPECTED_PUBLIC_API` tuple in the test, which in turn requires an ADR amendment (ADR-0001 Part (i) / ADR-0003). `segpaste._internal` is the reserved namespace for private modules P1 re-homes during the dense-sample migration.

## Testing

- `tests/test_copy_paste_fuzz.py` and `tests/test_placement_fuzz.py` use Hypothesis; expect longer runtimes than unit tests.
- `tests/shared.py` provides canonical LSJ / resize transform pipelines — reuse these rather than rebuilding similar pipelines in new tests.
- `tests/test_public_surface.py` enforces the ADR-pinned top-level API; see §Public surface.
- `pytest` is configured with `--strict-markers --strict-config` and coverage always on; don't add ad-hoc markers without registering them.
