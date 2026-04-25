# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project is pre-1.0; per the SemVer disclaimer in `README.md`, breaking
changes may land in any minor release.

## [0.3.0] — Unreleased

GPU-resident augmentation lane (W5, [ADR-0008](docs/adrs/0008-batch-copy-paste.md)).
Single-commit hard-deprecation of the entire CPU augmentation stack per
[ADR-0003](docs/adrs/0003-hard-deprecation-stance.md).

### Added

- `segpaste.BatchCopyPaste` — graph-compilable `nn.Module`
  (`torch.compile(fullgraph=True)` clean) that subsumes instance,
  panoptic, depth-aware, and classmix paste under one GPU pipeline:
  `BatchedPlacementSampler` → `AffinePropagator` → `TileCompositor`.
- `segpaste.PaddedBatchedDenseSample` — `[B, K, ...]` sibling of
  `BatchedDenseSample`, gated by `instance_valid`. Constructed via
  `BatchedDenseSample.to_padded(max_instances)`.
- `DenseSample.metric_depth: bool` paired with `camera_intrinsics`
  (ADR-0007 §1).
- `segpaste.integrations.huggingface` — pure-torch Mask2Former
  encoder/decoder for `{mask_labels, class_labels}` boundary
  (ADR-0006 §6).
- Compile-clean CI gate: `scripts/compile_explain.py` +
  `scripts/compile_allowlist.txt` (empty at v0.3.0;
  `tests/test_compile_clean.py` enforces).
- KS soft-report harness: `tests/test_batch_copy_paste_ks.py` records
  per-modality distances vs. `tests/fixtures/ks_snapshot.pt` as a CI
  artifact (no assertion at v0.3.0; ADR-0008 §D6 burn-in).
- GPU throughput lane: `benchmarks/bench_batch_copy_paste.py`,
  `.github/workflows/bench-gpu.yml` (`workflow_dispatch` only at
  v0.3.0), `benchmarks/compare.py --mode speedup-vs-cpu-baseline`.
- ADR-0005 (DenseComposite), ADR-0006 (PanopticPaste), ADR-0007
  (DepthAwarePaste), ADR-0008 (BatchCopyPaste).

### Removed

- `segpaste.CopyPasteCollator`, `segpaste.CopyPasteAugmentation`,
  `segpaste.CopyPasteConfig` — replaced by `BatchCopyPaste`.
- `InstancePaste`, `PanopticPaste`, `DepthAwarePaste`, `ClassMix` —
  collapsed into the single `BatchCopyPaste` forward.
- `segpaste.types.DetectionTarget` — already internal-only since
  v0.2.0; deleted now that every transform consumes `DenseSample`.
- CPU `PlacementSampler` and Protocol-based `processing/placement.py`
  — replaced by `_internal/gpu/batched_placement.py`.
- Per-wrapper parity fixtures and parity tests.
- Per-wrapper CPU benches; `benchmarks/_fixture.py` ragged builder
  replaced by `benchmarks/_padded_fixture.py`.
- `examples/` directory (entries depended on the deleted collator).

### Changed

- `pyproject.toml`: `torch` pin retained at `>=2.8`; the compile-clean
  allow-list is governed by ADR-0008 §D7 — additions to the allow-list
  require an ADR amendment.
- ADR-0001, ADR-0002, ADR-0003 amended to record the W5 hard-deprecation
  and the GPU dispatch lane (ADR-0002 Part v).

## [0.2.0]

Breaking release that closes the pre-1.0 public surface ahead of P1
(dense-sample composites). See [ADR-0003](docs/adrs/0003-hard-deprecation-stance.md)
for the deprecation stance; see [ADR-0001](docs/adrs/0001-dense-sample.md) for
the dense-sample surface.

### Removed

- `segpaste.CopyPasteTransform` — deleted outright; use
  `segpaste.CopyPasteCollator` or call
  `segpaste.CopyPasteAugmentation.transform(...)` directly. Pin
  `segpaste<0.9` if the old entry point is required.
- `segpaste.types.BoundingBox` — orphaned dataclass; not used internally and
  never surfaced in `segpaste.__all__`.
- `CopyPasteConfig.blend_mode` values `"gaussian"` and `"poisson"` — the
  wiring never existed. The `Literal` is tightened to `["alpha"]`; the two
  names remain **reserved** for future additive re-introduction via ADR.

### Changed

- `segpaste.DetectionTarget` — removed from `segpaste.__all__` and
  `segpaste.types.__all__`. The class is retained as an internal container
  for the duration of 0.9.0 because P1's W1 workstream still threads it
  through the augmentation pipeline; it is scheduled for removal in 0.9.1.
  External consumers migrate to `segpaste.DenseSample`; conversion is
  available via `DenseSample.from_detection_target` /
  `DenseSample.to_detection_target`.
- `segpaste.integrations.labels_getter` — demoted to the private
  `_internal` surface; removed from `segpaste.integrations.__all__`. Still
  importable by fully-qualified path for internal use.
- `segpaste.create_coco_dataloader` — promoted to the top-level public
  surface per ADR-0001 Part (i).

### Added

- `segpaste.__version__` — read from installed package metadata via
  `importlib.metadata.version`.
- `segpaste._internal` — explicit private namespace marker. P1 re-homes
  private modules here.
- `docs/adrs/0003-hard-deprecation-stance.md` — supersedes ADR-0001 Part (i)
  deprecation clauses.
- `tests/test_public_surface.py` — enforces `segpaste.__all__` against the
  ADR-pinned list, asserts no stray top-level names, and `xfail`s the
  `DetectionTarget` removal test pending P1.
- Coverage gate pinned under `[tool.coverage.report]` `fail_under`.
- `py.typed` presence and public-symbol-leak assertions added to the
  `publish.yml` wheel + sdist smoke tests.
