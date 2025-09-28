import torch
from torchvision.transforms import v2

from segpaste.augmentation import make_large_scale_jittering
from segpaste.augmentation.lsj import SanitizeBoundingBoxes
from segpaste.integrations import labels_getter


def generate_scale_jitter_transform_strategy(
    min_scale: float = 0.1, max_scale: float = 2.0
) -> v2.Transform:
    return v2.Compose(
        [
            v2.ToImage(),
            v2.RandomHorizontalFlip(),
            make_large_scale_jittering(
                output_size=(256, 256), min_scale=min_scale, max_scale=max_scale
            ),
            v2.ClampBoundingBoxes(),
            SanitizeBoundingBoxes(labels_getter=labels_getter),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )


def generate_resize_transform_strategy() -> v2.Transform:
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(size=(256, 256)),
            v2.RandomHorizontalFlip(),
            v2.ClampBoundingBoxes(),
            SanitizeBoundingBoxes(labels_getter=labels_getter),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
