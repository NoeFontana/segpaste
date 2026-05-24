# Decision records

Every architectural decision is recorded as an Architecture Decision
Record (ADR). The table below maps the active subsystems to the ADRs
that govern them; the navigation sidebar lists every ADR in order.

## By subsystem

| Subsystem                                          | Governing ADRs                                                                                                          |
| -------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Type system (`DenseSample`, modality matrix)       | [0001](0001-dense-sample.md), [0004](0004-batched-dense-sample.md)                                                       |
| Stability contract (public surface, deprecation)   | [0001](0001-dense-sample.md) Part (i), [0003](0003-hard-deprecation-stance.md), [0015](0015-thirty-minute-integration.md) |
| `BatchCopyPaste` GPU kernel + `torch.compile`      | [0008](0008-batch-copy-paste.md)                                                                                        |
| Per-modality composite primitives                  | [0005](0005-dense-composite.md), [0006](0006-panoptic-paste.md), [0007](0007-depth-aware-paste.md)                       |
| Patch-aligned paste (ViT backbones)                | [0010](0010-patch-aligned-paste.md)                                                                                     |
| Image harmonization (Reinhard / multi-band / Poisson) | [0012](0012-image-harmonization.md)                                                                                  |
| Source strategy + InstanceBank                     | [0011](0011-instance-bank.md)                                                                                           |
| Presets registry                                   | [0009](0009-visual-validation-and-presets.md) §3, [0015](0015-thirty-minute-integration.md)                              |
| Visual validation + FiftyOne substrate             | [0009](0009-visual-validation-and-presets.md), [0013](0013-fiftyone-visualizer-substrate.md)                             |
| Audit sidecar (invariant dispatch)                 | [0014](0014-batchauditpacket-forward-return-sidecar.md), [0001](0001-dense-sample.md) Part (ii)                          |
| Performance baseline                               | [0002](0002-performance-baseline.md)                                                                                    |
| Framework adapters (torchvision / HF / Lightning)  | [0015](0015-thirty-minute-integration.md)                                                                               |

## ADR workflow

This is a solo + agentic project (per
[`MEMORY.md`](https://github.com/NoeFontana/segpaste/blob/main/CLAUDE.md)).
Every change to the codebase that touches public API, type-system
shape, or any of the governed subsystems above lands together with the
ADR that records the decision. No experimental namespace, no runtime
compat shims, no review-gate framing — the ADR *is* the contract.

ADRs are append-only once Accepted. Subsequent ADRs may amend or
supersede earlier ones; supersession is recorded in the header table.
