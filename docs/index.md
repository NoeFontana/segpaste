# SegPaste

A PyTorch reimplementation of
["Simple Copy-Paste is a Strong Data Augmentation Method for Instance
Segmentation"](https://arxiv.org/abs/2012.07177). One GPU-resident,
`torch.compile(fullgraph=True)`-clean `nn.Module` covering instance,
panoptic, depth-aware, and class-mix copy-paste under a single
preset-driven API.

## Where to go

- **[Getting started](getting-started.md)** — install, look up a
  preset, run one training step against Hugging Face Mask2Former.
- **[Migration guide](migration.md)** — swap in from
  `torchvision.transforms.v2.SimpleCopyPaste` or
  `mmdet.datasets.transforms.CopyPaste`.
- **[Design principles](design-principles.md)** — what segpaste is
  opinionated about, with pointers to the governing ADRs.
- **[Reference](reference/api.md)** — pinned public API,
  [presets](reference/presets.md), and
  [framework adapters](reference/integrations.md).
- **[Decision records](adrs/index.md)** — every architectural decision
  is recorded as an ADR.

## Status

Pre-1.0. The public API in `segpaste.__all__` is subject to breaking
changes; ADR-0003 records the hard-deprecation stance (no compat
shims). See the [design principles](design-principles.md) page for
what this means for downstream users.
