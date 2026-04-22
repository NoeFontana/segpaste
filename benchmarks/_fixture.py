"""Synthetic batch builder for the `CopyPasteCollator` benchmark.

Sole source of truth for the workload shape pinned in ADR-0002.
"""

import torch
from torchvision import tv_tensors

from segpaste.types import DenseSample, InstanceMask


def build_batch(
    seed: int,
    batch_size: int = 8,
    img_size: int = 1024,
    k_range: tuple[int, int] = (1, 5),
) -> list[DenseSample]:
    """Build one synthetic batch shaped for `CopyPasteCollator.__call__`.

    Each sample is a :class:`DenseSample` in INSTANCE modality. ``k`` objects
    per sample are drawn uniformly from ``[k_range[0], k_range[1]]``
    (inclusive). Masks are the interior of each box (bool), which guarantees
    ``area >= min_object_area=1`` for the canonical benchmark config.
    """
    g = torch.Generator().manual_seed(seed)
    k_lo, k_hi = k_range
    batch: list[DenseSample] = []

    for i in range(batch_size):
        gi = torch.Generator().manual_seed(seed * 8191 + i)

        image_tensor = torch.randint(
            0, 256, (3, img_size, img_size), generator=gi, dtype=torch.uint8
        )
        image = tv_tensors.Image(image_tensor)

        k = int(torch.randint(k_lo, k_hi + 1, (1,), generator=g).item())

        w = torch.randint(64, 257, (k,), generator=gi, dtype=torch.int64)
        h = torch.randint(64, 257, (k,), generator=gi, dtype=torch.int64)
        x1 = torch.randint(0, img_size - 256, (k,), generator=gi, dtype=torch.int64)
        y1 = torch.randint(0, img_size - 256, (k,), generator=gi, dtype=torch.int64)
        x2 = x1 + w
        y2 = y1 + h

        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1).to(torch.float32)
        boxes = tv_tensors.BoundingBoxes(  # pyright: ignore[reportCallIssue]
            boxes_xyxy,
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=(img_size, img_size),
        )

        labels = torch.randint(1, 81, (k,), generator=gi, dtype=torch.int64)

        masks = torch.zeros((k, img_size, img_size), dtype=torch.bool)
        for j in range(k):
            masks[j, int(y1[j]) : int(y2[j]), int(x1[j]) : int(x2[j])] = True

        batch.append(
            DenseSample(
                image=image,
                boxes=boxes,
                labels=labels,
                instance_masks=InstanceMask(masks),
                instance_ids=torch.arange(k, dtype=torch.int32),
            )
        )

    return batch
