# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project is pre-1.0; per the SemVer disclaimer in `README.md`, breaking
changes may land in any minor release.

## [0.9.0] — Unreleased

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
