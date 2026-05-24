# Integrations

segpaste ships a small set of framework adapters in
`src/segpaste/integrations/`. Each is intentionally thin — the goal is
to get a registered preset into the framework's native training loop in
under 30 minutes.

=== "COCO"

    `CocoDetectionV2` and `CocoPanopticV2` are
    `torchvision.transforms.v2`-compatible dataset loaders that yield
    `DenseSample` objects. `create_coco_dataloader` is the matching
    `DataLoader` factory.

    ::: segpaste.integrations.coco.CocoDetectionV2

    ::: segpaste.integrations.coco.CocoPanopticV2

    ::: segpaste.create_coco_dataloader

=== "torchvision"

    `make_segpaste_collate_fn` compresses the
    `BatchedDenseSample.from_samples(...).to_padded(K)` ritual into a
    single `collate_fn`. Wire it into a stock `DataLoader`, then run
    `BatchCopyPaste(get_preset(name).batch_copy_paste)` on the
    resulting padded batch.

    ::: segpaste.integrations.torchvision.make_segpaste_collate_fn

=== "Hugging Face"

    `to_hf_format` / `from_hf_format` convert a single `DenseSample`
    to and from the
    `{mask_labels, class_labels, pixel_values}` dict shape consumed by
    `transformers.Mask2FormerImageProcessor`. `to_hf_batch` is the
    batch-level pairing; `make_hf_collate_fn` closes the loop end-to-end
    inside a `DataLoader.collate_fn`. No `transformers` import — the
    adapter is structurally compatible.

    ::: segpaste.integrations.huggingface.to_hf_format

    ::: segpaste.integrations.huggingface.from_hf_format

    ::: segpaste.integrations.huggingface.to_hf_batch

    ::: segpaste.integrations.huggingface.make_hf_collate_fn

=== "Lightning"

    `make_segpaste_datamodule` is a factory returning a preset-bound
    `LightningDataModule`. Augmentation runs in
    `on_after_batch_transfer` — on the GPU side of the device transfer,
    matching `BatchCopyPaste`'s compile-clean contract (ADR-0008 §D7).

    `lightning` is an optional dependency:
    `pip install 'segpaste[lightning]'`. Calling the factory without
    `lightning` installed raises `ImportError` with the install hint.

    ::: segpaste.integrations.lightning.make_segpaste_datamodule
