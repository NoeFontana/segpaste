# Presets

Registered presets bind per-dataset `BatchCopyPasteConfig` defaults.
Look one up with `segpaste.get_preset(name)`; list the registered names
with `segpaste.list_presets()`. The two that ship today both follow the
COCO eval ergonomics described in ADR-0009 §3 and target the v0.3.0
`BatchCopyPaste` kernel.

A preset is a frozen Pydantic model (`PresetConfig`) carrying the
augmentation hyperparameters, the modality set the preset expects, and
an optional `SignOff` audit trail.

## `coco-instance`

Instance segmentation copy-paste, paper-aligned. LSJ scale range
`(0.5, 2.0)`, 50% horizontal flip, 32-instance paste-count cap, and a
10% residual-area drop that matches COCO eval's
"discard ≥ 90%-occluded annotations" convention.

::: segpaste.presets.coco_instance.COCO_INSTANCE

## `coco-panoptic`

Panoptic copy-paste with Mask2Former-inspired density
(`paste_prob=0.5`, `k_range=(1, 20)`), thing-only paste source per
ADR-0006 §2, and a `tau_stuff_frac=0.1` remaining-area floor on every
stuff class. The 133-entry COCO panoptic taxonomy is bundled as a
numeric thing/stuff mapping; no class name strings are vendored
(ADR-0009 §1).

::: segpaste.presets.coco_panoptic.COCO_PANOPTIC

::: segpaste.presets.coco_panoptic.COCO_PANOPTIC_SCHEMA

## Building your own

```python
from segpaste import PresetConfig, register_preset
from segpaste.augmentation.batch_copy_paste import BatchCopyPasteConfig
from segpaste.types import Modality

MY_PRESET = PresetConfig(
    name="my-dataset",
    description="...",
    batch_copy_paste=BatchCopyPasteConfig(...),
    target_modalities=(Modality.IMAGE, Modality.INSTANCE),
)
register_preset(MY_PRESET)
```

Registration is process-local and at import-time. The repo's two
presets register themselves through side-effect imports in
`src/segpaste/presets/__init__.py`; downstream packages can do the same.
