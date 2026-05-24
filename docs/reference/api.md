# Public API

Pinned by `tests/test_public_surface.py::test_all_matches_pinned_surface`.
Adding or removing a name from this page requires amending both
`segpaste.__all__` and the `_EXPECTED_PUBLIC_API` tuple in the test,
which in turn requires an ADR amendment (ADR-0001 Part (i) / ADR-0003).

The full surface as of v0.3.x is grouped below; see also the dedicated
[Presets](presets.md) and [Integrations](integrations.md) pages for
prose context.

## Augmentation

::: segpaste.BatchCopyPaste

::: segpaste.SourceStrategy

::: segpaste.IntraBatchSource

::: segpaste.BankSource

::: segpaste.InstanceBank

::: segpaste.make_large_scale_jittering

::: segpaste.FixedSizeCrop

::: segpaste.RandomResize

::: segpaste.SanitizeBoundingBoxes

## Types

::: segpaste.DenseSample

::: segpaste.BatchedDenseSample

::: segpaste.PaddedBatchedDenseSample

::: segpaste.Modality

::: segpaste.InstanceMask

::: segpaste.SemanticMap

::: segpaste.PanopticMap

::: segpaste.PanopticSchema

::: segpaste.PaddingMask

::: segpaste.CameraIntrinsics

::: segpaste.BatchAuditPacket

## Presets

::: segpaste.PresetConfig

::: segpaste.get_preset

::: segpaste.list_presets

::: segpaste.register_preset

## Integrations

### COCO

::: segpaste.CocoDetectionV2

::: segpaste.create_coco_dataloader

### Framework adapters

::: segpaste.make_segpaste_collate_fn

::: segpaste.to_hf_format

::: segpaste.from_hf_format

::: segpaste.to_hf_batch

::: segpaste.make_hf_collate_fn

::: segpaste.make_segpaste_datamodule
