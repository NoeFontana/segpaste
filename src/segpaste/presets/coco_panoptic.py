"""COCO panoptic segmentation preset (ADR-0009 §3, P6).

Pins the panoptic-aware :class:`BatchCopyPasteConfig` defaults:
Mask2Former-inspired density (``paste_prob=0.5``, ``k_range=(1, 20)``),
thing-only paste source (ADR-0006 §2), and ``tau_stuff_frac=0.1``
remaining-area floor (ADR-0006 §3 partial). Importing this module
registers the preset under the name ``"coco-panoptic"``.

The 133-entry class taxonomy below mirrors the canonical
``panoptic_coco_categories.json`` IDs and is a numeric mapping only —
no class name strings are bundled, keeping the preset license-clean
per ADR-0009 §1.
"""

from __future__ import annotations

from typing import Literal

from segpaste._internal.gpu.batched_placement import BatchedPlacementConfig
from segpaste._internal.gpu.tile_composite import TileCompositorConfig
from segpaste.augmentation.batch_copy_paste import (
    BatchCopyPasteConfig,
    PanopticPasteConfig,
)
from segpaste.presets import register_preset
from segpaste.presets._base import PresetConfig
from segpaste.types import Modality, PanopticSchemaSpec

COCO_PANOPTIC_CLASSES: dict[int, Literal["thing", "stuff"]] = {
    1: "thing",
    2: "thing",
    3: "thing",
    4: "thing",
    5: "thing",
    6: "thing",
    7: "thing",
    8: "thing",
    9: "thing",
    10: "thing",
    11: "thing",
    13: "thing",
    14: "thing",
    15: "thing",
    16: "thing",
    17: "thing",
    18: "thing",
    19: "thing",
    20: "thing",
    21: "thing",
    22: "thing",
    23: "thing",
    24: "thing",
    25: "thing",
    27: "thing",
    28: "thing",
    31: "thing",
    32: "thing",
    33: "thing",
    34: "thing",
    35: "thing",
    36: "thing",
    37: "thing",
    38: "thing",
    39: "thing",
    40: "thing",
    41: "thing",
    42: "thing",
    43: "thing",
    44: "thing",
    46: "thing",
    47: "thing",
    48: "thing",
    49: "thing",
    50: "thing",
    51: "thing",
    52: "thing",
    53: "thing",
    54: "thing",
    55: "thing",
    56: "thing",
    57: "thing",
    58: "thing",
    59: "thing",
    60: "thing",
    61: "thing",
    62: "thing",
    63: "thing",
    64: "thing",
    65: "thing",
    67: "thing",
    70: "thing",
    72: "thing",
    73: "thing",
    74: "thing",
    75: "thing",
    76: "thing",
    77: "thing",
    78: "thing",
    79: "thing",
    80: "thing",
    81: "thing",
    82: "thing",
    84: "thing",
    85: "thing",
    86: "thing",
    87: "thing",
    88: "thing",
    89: "thing",
    90: "thing",
    92: "stuff",
    93: "stuff",
    95: "stuff",
    100: "stuff",
    107: "stuff",
    109: "stuff",
    112: "stuff",
    118: "stuff",
    119: "stuff",
    122: "stuff",
    125: "stuff",
    128: "stuff",
    130: "stuff",
    133: "stuff",
    138: "stuff",
    141: "stuff",
    144: "stuff",
    145: "stuff",
    147: "stuff",
    148: "stuff",
    149: "stuff",
    151: "stuff",
    154: "stuff",
    155: "stuff",
    156: "stuff",
    159: "stuff",
    161: "stuff",
    166: "stuff",
    168: "stuff",
    171: "stuff",
    175: "stuff",
    176: "stuff",
    177: "stuff",
    178: "stuff",
    180: "stuff",
    181: "stuff",
    184: "stuff",
    185: "stuff",
    186: "stuff",
    187: "stuff",
    188: "stuff",
    189: "stuff",
    190: "stuff",
    191: "stuff",
    192: "stuff",
    193: "stuff",
    194: "stuff",
    195: "stuff",
    196: "stuff",
    197: "stuff",
    198: "stuff",
    199: "stuff",
    200: "stuff",
}

COCO_PANOPTIC_SCHEMA = PanopticSchemaSpec(
    classes=COCO_PANOPTIC_CLASSES,
    ignore_index=255,
    max_instances_per_image=256,
)

COCO_PANOPTIC = PresetConfig(
    name="coco-panoptic",
    description=(
        "COCO panoptic segmentation copy-paste. Mask2Former-inspired "
        "density (paste_prob=0.5, k_range=(1, 20)); thing-only paste "
        "source per ADR-0006 §2; tau_stuff_frac=0.1 enforces a "
        "minimum remaining-area fraction per stuff class."
    ),
    batch_copy_paste=BatchCopyPasteConfig(
        placement=BatchedPlacementConfig(
            scale_range=(0.5, 2.0),
            hflip_probability=0.5,
            paste_prob=0.5,
            k_range=(1, 20),
        ),
        composite=TileCompositorConfig(tile_size=512),
        min_residual_area_frac=0.1,
        panoptic=PanopticPasteConfig(
            taxonomy=COCO_PANOPTIC_SCHEMA,
            tau_stuff_frac=0.1,
        ),
    ),
    target_modalities=(
        Modality.IMAGE,
        Modality.INSTANCE,
        Modality.SEMANTIC,
        Modality.PANOPTIC,
    ),
)

register_preset(COCO_PANOPTIC)
