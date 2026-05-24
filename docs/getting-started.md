# Getting started

A pre-trained Hugging Face Mask2Former, a registered preset, and one
training step — in under 50 lines.

## What you get

`segpaste.BatchCopyPaste` is one `nn.Module` that runs the
Copy-Paste augmentation from [Ghiasi et al. 2020](https://arxiv.org/abs/2012.07177)
on the GPU. It consumes a `PaddedBatchedDenseSample` (a leading-batch
dataclass of tensors) and returns one — every modality (image, instance
masks, panoptic map, depth, normals) flows through a single
`torch.compile(fullgraph=True)`-clean forward (ADR-0008).

Registered presets bind the per-dataset hyperparameters. The two that
ship today are `coco-instance` and `coco-panoptic` (ADR-0009 §3); list
the current set with `segpaste.list_presets()`.

## Install

```bash
pip install segpaste
pip install 'transformers>=5.9'    # for the Mask2Former example below
pip install 'segpaste[lightning]'  # only if you use the Lightning adapter
```

Python 3.11+, PyTorch ≥ 2.8, torchvision ≥ 0.23 are required.

## The 50-line training step

```python
import torch
from torch.utils.data import DataLoader
from transformers import Mask2FormerForUniversalSegmentation

from segpaste import (
    BatchCopyPaste,
    BatchedDenseSample,
    get_preset,
)
from segpaste.integrations.coco import CocoPanopticV2
from segpaste.integrations.huggingface import to_hf_batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Point this at your COCO 2017 panoptic root.
dataset = CocoPanopticV2(
    image_folder="/path/to/coco/val2017",
    panoptic_folder="/path/to/coco/annotations/panoptic_val2017",
    label_path="/path/to/coco/annotations/panoptic_val2017.json",
)
loader = DataLoader(dataset, batch_size=2, collate_fn=list)

# 2. Look up the preset and build the augmenter (ADR-0009 §3).
preset = get_preset("coco-panoptic")
augment = BatchCopyPaste(preset.batch_copy_paste).to(device)

# 3. Build the Hugging Face model (panoptic Mask2Former head).
model = Mask2FormerForUniversalSegmentation.from_pretrained(
    "facebook/mask2former-swin-tiny-coco-panoptic"
).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# 4. One training step end-to-end.
for samples in loader:
    padded = BatchedDenseSample.from_samples(samples).to_padded(max_instances=32)
    padded = padded.to(device)  # pyright: ignore[reportAttributeAccessIssue]
    padded = augment(padded, generator=torch.Generator(device=device))
    hf = to_hf_batch(padded)

    outputs = model(
        pixel_values=hf["pixel_values"],
        mask_labels=hf["mask_labels"],
        class_labels=hf["class_labels"],
    )
    outputs.loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    break
```

## What just happened

1. `CocoPanopticV2` produces per-sample `DenseSample` instances with
   `instance_masks`, `labels`, `semantic_map`, and `panoptic_map` already
   populated.
2. `BatchedDenseSample.from_samples(...)` collates ragged per-sample
   instance counts (ADR-0004). `.to_padded(K)` packs them into a fully
   rectangular `PaddedBatchedDenseSample` consumable by
   `torch.compile(fullgraph=True)` (ADR-0008).
3. `get_preset("coco-panoptic")` returns a frozen `PresetConfig` pinning
   the Mask2Former-inspired density (paste_prob=0.5, k_range=(1, 20)),
   thing-only paste source (ADR-0006 §2), and `tau_stuff_frac=0.1`
   remaining-area floor.
4. `BatchCopyPaste(...)` is the public entry point that replaces every
   pre-v0.3.0 CPU collator and modality wrapper with a single
   GPU-resident kernel (ADR-0008).
5. `to_hf_batch(padded)` converts the post-augmentation batch into
   `{mask_labels: list[Tensor], class_labels: list[Tensor], pixel_values: Tensor}`
   — the training-time forward signature of
   `Mask2FormerForUniversalSegmentation`.

## Next

- Switching from `torchvision.transforms.v2.SimpleCopyPaste` or
  `mmdet.datasets.transforms.CopyPaste`? See the
  [migration guide](migration.md).
- Why segpaste is preset-based and not transform-chained: see
  [design principles](design-principles.md).
- Full API surface: [reference/api.md](reference/api.md).
- Lightning users: skip the manual loop and use
  `make_segpaste_datamodule` from
  [reference/integrations.md](reference/integrations.md).
