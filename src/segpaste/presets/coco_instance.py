"""COCO instance segmentation preset (ADR-0009 §3, P5).

Pins LSJ-aligned :class:`BatchCopyPasteConfig` defaults derived from the
paper recipe and the v0.3.0 ergonomics: scale range ``(0.5, 2.0)``, hflip
50%, 32-instance paste-count cap, 10% residual-area drop. Importing this
module registers the preset under the name ``"coco-instance"``.
"""

from __future__ import annotations

from segpaste._internal.gpu.batched_placement import BatchedPlacementConfig
from segpaste._internal.gpu.tile_composite import TileCompositorConfig
from segpaste.augmentation.batch_copy_paste import BatchCopyPasteConfig
from segpaste.presets import register_preset
from segpaste.presets._base import PresetConfig
from segpaste.types import Modality

COCO_INSTANCE = PresetConfig(
    name="coco-instance",
    description=(
        "COCO instance segmentation copy-paste, paper-aligned. "
        "LSJ scale_range=(0.5, 2.0), hflip 50%, 32-instance paste-count "
        "cap, and 10% residual-area drop matching COCO eval ergonomics."
    ),
    batch_copy_paste=BatchCopyPasteConfig(
        placement=BatchedPlacementConfig(
            scale_range=(0.5, 2.0),
            hflip_probability=0.5,
            paste_prob=1.0,
            k_range=(1, 32),
        ),
        composite=TileCompositorConfig(tile_size=512),
        min_residual_area_frac=0.1,
    ),
    target_modalities=(Modality.IMAGE, Modality.INSTANCE),
)

register_preset(COCO_INSTANCE)
