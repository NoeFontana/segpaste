# ADR-0011 — InstanceBank: external class-balanced source pool (A1)

|            |                                                                                                                          |
| ---------- | ------------------------------------------------------------------------------------------------------------------------ |
| Number     | 0011                                                                                                                     |
| Title      | `InstanceBank` protocol + `BankSource` strategy: external class-balanced crop pool for `BatchCopyPaste`                  |
| Status     | Accepted                                                                                                                 |
| Author     | @NoeFontana                                                                                                              |
| Created    | 2026-05-10                                                                                                               |
| Tag        | `ADR-0011`                                                                                                               |
| Relates-to | [ADR-0001](0001-dense-sample.md) Part (i) (public surface), Part (iv) (RNG); [ADR-0008](0008-batch-copy-paste.md) §D7 (compile-clean) |
| Amends     | [ADR-0008](0008-batch-copy-paste.md) §C4 — `AffinePropagator.forward` is now `(target, source, placement)`, taking the source view from the new `SourceStrategy.sample` boundary |

## Context

`BatchCopyPaste` v0.3.0 picks paste sources from the same batch via a
diagonal-masked `torch.multinomial` over a `[B, B]` weights matrix
(`_internal/gpu/batched_placement.py`). The empirical class distribution
of pasted instances therefore matches the dataset's natural class skew —
there is no mechanism to over-sample rare classes.

The X-Paste result (+6.5 AP_r on LVIS, [arXiv:2212.03863](https://arxiv.org/abs/2212.03863))
shows that biasing the source distribution toward under-represented
classes closes the rare-class gap on long-tail benchmarks. Any robotics
class distribution where `max f_c / min f_c ≳ 10³` benefits from the
same lever.

## Decision

Three layers, all gated by this ADR:

1. A polymorphic source-selection seam at the input to
   `AffinePropagator`. `SourceStrategy.sample(target, valid_extent,
   source_eligible, generator) -> (source_view, placement)` — the
   `source_view` is row-aligned with `target` along the batch dim, and
   `placement.source_idx` indexes into the source view. Default
   `IntraBatchSource()` returns `target` itself with the diagonal-masked
   multinomial — bitwise identical to v0.3.0.
2. An external `InstanceBank` Protocol with three concrete backends —
   `MemmapBank` (in-process hot-path, `np.memmap`), `LMDBBank`
   (single-host NVMe, B-tree via `lmdb`), `WebDatasetBank` (sharded tar
   + parquet index, multi-TB streaming). All backends carry the same
   on-disk metadata: per-crop `(image, alpha, class_id, optional
   embedding)` plus `meta.json#sha256` for cache keys. A separate
   `BankSampler(torch.utils.data.Sampler)` implements Gupta 2019
   repeat-factor weighting `r(c) = max(1, sqrt(t / f_c))` with
   `(base_seed, epoch, rank)` SHA-256 mixing per ADR-0001 Part (iv).
3. `BankSource(SourceStrategy)` consumes a per-step pre-staged tensor
   `[B, K_bank, C+2, h, w]` (RGB + alpha + broadcast class-id), drawn
   class-balanced in the `BankSampler` and packed by the worker collate
   `stage_bank_batch`. The strategy multinomial-picks one bank crop per
   target row, places it at the origin of a target-canvas-sized
   synthetic source view (`K_source = 1`), and draws per-target
   `(scale, translate, hflip)` parameters. `placement.source_idx =
   arange(B)`.

`AffinePropagator.forward` becomes `forward(target, source, placement)`
and reads canvas geometry (`B`, `K`, `H`, `W`, `device`) from `target`
while gathering every modality from `source.x[source_idx]`. The output
declares `max_instances = source.max_instances` so its tensor shapes
match its declared K, regardless of which strategy emits the source view.

### Public surface adds (ADR-0001 Part (i) amendment)

- `SourceStrategy` — runtime-checkable Protocol
- `IntraBatchSource` — v0.3.0-equivalent default
- `BankSource` — external-bank source
- `InstanceBank` — bank Protocol

`MemmapBank`, `LMDBBank`, `WebDatasetBank`, `BankSampler`,
`stage_bank_batch`, and `create_bank_dataloader` stay under
`segpaste._internal.bank` until a follow-up ADR amendment promotes
them, matching the project convention for unvalidated GPU primitives
(ADR-0005 / ADR-0008).

### Compile-clean preservation

Every operation that could break the empty `scripts/compile_allowlist.txt`
lives **outside** `BatchCopyPaste.forward`:

- Disk I/O / decode / RLE expansion → `bank.__getitem__` (worker process)
- Class-balanced index sampling → `BankSampler.__iter__` (worker)
- Predicate filtering (e.g. CLIP threshold) → `BankCocoCropDataset`
  (worker, deterministic walk on rejection)
- Crop staging → `stage_bank_batch` (worker collate, single
  pre-allocated buffer + scatter)

The forward sees only:
- The target `PaddedBatchedDenseSample` (unchanged)
- A pre-staged source-view tensor produced from the bank batch
- One `torch.multinomial` over a uniform weight tensor to pick which
  staged candidate fires per target row

`tests/test_bank_compile_clean.py` parametrizes the existing
ADR-0008 §D7 gate over the BankSource path — empty allow-list passes.
`tests/test_batch_copy_paste_bitwise.py` pins
`BatchCopyPaste(default_config)` against the v0.3.0 forward output;
the default `IntraBatchSource` is byte-identical.

### Bank serialization

All three backends share the same logical record:

```
image:      uint8  [3, h, w]   (RGB, no decode at read time)
alpha:      bool   [1, h, w]   (instance mask, decoded RLE)
class_id:   int64               (zero-indexed)
embedding:  float16 [256] | None (optional CLIP-style vector)
```

Per-backend layout:

| Backend | Layout |
|---|---|
| `memmap` | `images.dat` + `alpha.dat` + `classes.npy` + optional `embeddings.dat` + `meta.json` |
| `lmdb` | `data.mdb` (B-tree, packed binary records) + `meta.json` |
| `webdataset` | `shard-{NNNNNN}.tar` (`crop_{idx:08d}.bin` packed records) + `index.parquet` + `meta.json` |

`meta.json` carries `format`, `format_version`, `num_crops`, `crop_h`,
`crop_w`, `num_classes`, `has_embeddings`, `segpaste_version`,
`build_seed`, `class_frequencies`, and a SHA-256 of itself (with the
hash field excluded) used as the `version` cache key.

### Optional dependencies

- `bank-memmap`: `numpy >= 1.26`
- `bank-lmdb`: `lmdb >= 1.4`
- `bank-webdataset`: `webdataset >= 0.2.86`, `pyarrow >= 15`
- `bank` (umbrella): all three above

Each backend's import is guarded by a `require_*` helper in
`segpaste._internal.imports` that raises `ImportError` with an
install hint at first use.

### Build path

`scripts/build_instance_bank.py` walks COCO instances in
`(image_id, annotation_id)` lexicographic order, crops each annotation
to its bbox, center-pads to `--crop-size`, and writes the bank in the
selected format. Two builds with the same inputs produce byte-identical
banks (verified by `meta.json#sha256`).

### KS soft-report extension

`tests/test_batch_copy_paste_ks.py` already records paste-area /
per-image paste count / per-class histograms over `IntraBatchSource`
draws (ADR-0008 §D6 30-day burn-in). After this ADR, the test is
parametrized over `["intrabatch", "bank"]`; the bank-mode snapshot
lives at `tests/fixtures/ks_snapshot_bank.pt` (added in a follow-up
once the bank-mode burn-in begins).

## Consequences

- The default-config forward path is bitwise-identical to v0.3.0;
  existing pipelines do not need to change.
- A new `source: SourceConfig` field on `BatchCopyPasteConfig`
  (discriminated by `kind`) plumbs both strategies through Pydantic
  config — YAML configs without `source:` continue to load unchanged.
- `K_source = 1` (one paste per target slot) matches v0.3.0 "one
  source per target" semantics. Multi-crop-per-target bursts are a
  future extension; relaxing this lifts the source-view's K beyond 1
  and the slot-merge logic (`_merge_slots`) is already shaped for
  arbitrary `K_s` (verified by the bank-mode tests with `K_s = 1`
  but the merge code uses `K_t` for output K).
- `AffinePropagator.forward` signature change from `(padded,
  placement)` to `(target, source, placement)` is a hard break for
  any direct caller. The repo's only direct callers (`BatchCopyPaste`,
  `tests/test_affine_propagate.py`, `tests/test_batch_copy_paste_ks.py`)
  are migrated as part of this ADR.

## Tracking

- Initial PR sequence: PR1 propagator refactor → PR2 SourceStrategy
  protocol → PR3 SourceConfig discriminated union → PR4 InstanceBank
  + MemmapBank + BankSampler + loader → PR5 build script + LMDBBank →
  PR6 WebDatasetBank → PR7 BankSource + KS gate + this ADR.
- Burn-in: ADR-0008 §D6 30-day window applies to the bank-mode KS
  draws once a snapshot fixture lands. After the burn-in, a follow-up
  amendment promotes the bank-mode KS gate from soft-report to hard
  threshold.
- Follow-ups: predicate-filtered hot path (CLIP) instrumentation,
  multi-crop-per-target (`K_source > 1`), promotion of backend classes
  + `BankSampler` to the public surface.
