# Migration guide

segpaste is preset-based, batched, GPU-resident, and panoptic-aware
out of the box. The translations below preserve semantic intent
(copy-paste augmentation for instance / panoptic segmentation), not API
shape. Expect the diffs to look surface-level different — they are. The
underlying compute path is also different (post-collate, on-device,
graph-clean).

## From `torchvision.transforms.v2.SimpleCopyPaste`

=== "Before"

    ```python
    from torchvision.transforms import v2

    transform = v2.Compose([
        v2.ToImage(),
        v2.SimpleCopyPaste(
            blending=True,
            resize_interpolation=v2.InterpolationMode.BILINEAR,
            antialias=True,
        ),
        v2.ToDtype(torch.float32, scale=True),
    ])
    loader = DataLoader(dataset, batch_size=8, collate_fn=transform)
    ```

=== "After"

    ```python
    from segpaste import (
        BatchCopyPaste,
        BatchedDenseSample,
        get_preset,
        make_segpaste_collate_fn,
    )

    augment = BatchCopyPaste(get_preset("coco-instance").batch_copy_paste).to(device)
    loader = DataLoader(dataset, batch_size=8,
                        collate_fn=make_segpaste_collate_fn(max_instances=32))

    for padded in loader:
        padded = augment(padded.to(device))
    ```

!!! warning "Semantic difference"
    `torchvision.SimpleCopyPaste` runs inside the `collate_fn` on CPU
    and operates only on instance masks. segpaste's `BatchCopyPaste` is
    an `nn.Module` that runs *after* collate, on whatever device the
    batch is on, and is panoptic-aware (the `coco-panoptic` preset
    activates thing-only sources and a stuff-area-threshold revert per
    ADR-0006). The augmentation is also `torch.compile(fullgraph=True)`-
    clean.

## From `mmdet.datasets.transforms.CopyPaste`

=== "Before"

    ```python
    # In an mmdet config:
    train_pipeline = [
        dict(type="LoadImageFromFile"),
        dict(type="LoadAnnotations", with_mask=True),
        dict(type="CopyPaste", max_num_pasted=100),
        dict(type="PackDetInputs"),
    ]
    ```

=== "After"

    ```python
    from segpaste import BatchCopyPaste, get_preset
    from segpaste.augmentation.batch_copy_paste import BatchCopyPasteConfig

    # Start from a preset and override the k_range to mirror max_num_pasted.
    preset = get_preset("coco-instance")
    cfg = preset.batch_copy_paste.model_copy(
        update={"placement": preset.batch_copy_paste.placement.model_copy(
            update={"k_range": (1, 100)},
        )},
    )
    augment = BatchCopyPaste(cfg).to(device)
    ```

!!! warning "Semantic difference"
    mmdet's `CopyPaste` is one pipeline stage among many; the rest of
    the augmentation graph (resize, flip, normalize) lives in the same
    config. segpaste deliberately does **not** own that graph — use
    `segpaste.make_large_scale_jittering` as the dataset-side LSJ
    transform (it is a `torchvision.transforms.v2.Transform` and
    composes with the rest of the v2 ecosystem), and let
    `BatchCopyPaste` operate exclusively post-collate. The frozen
    `BatchCopyPasteConfig` (`extra="forbid"`) means overrides are
    explicit via `model_copy`; there is no per-call kwarg surface.
