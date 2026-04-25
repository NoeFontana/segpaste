"""Padded-batched fixture builder for the BatchCopyPaste bench (ADR-0008 §v).

Replaces the ragged ``_fixture.py`` that fed the deleted CPU wrappers. The
returned objects are :class:`PaddedBatchedDenseSample` instances so the
bench feeds the compilable path directly — no ragged Python fields cross
the ``torch.compile`` boundary.
"""

from __future__ import annotations

import random

import torch
from torchvision import tv_tensors

from segpaste import PaddedBatchedDenseSample
from segpaste.types import (
    BatchedDenseSample,
    DenseSample,
    InstanceMask,
)


def _sample(image_size: int, k: int, seed: int, device: torch.device) -> DenseSample:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    image_data = torch.rand(3, image_size, image_size, generator=gen)
    masks = torch.zeros(k, image_size, image_size, dtype=torch.bool)
    boxes: list[list[int]] = []
    for i in range(k):
        side = int(torch.randint(8, image_size // 4, (1,), generator=gen).item())
        x1 = int(torch.randint(0, image_size - side, (1,), generator=gen).item())
        y1 = int(torch.randint(0, image_size - side, (1,), generator=gen).item())
        masks[i, y1 : y1 + side, x1 : x1 + side] = True
        boxes.append([x1, y1, x1 + side, y1 + side])
    return DenseSample(
        image=tv_tensors.Image(image_data.to(device)),
        boxes=tv_tensors.BoundingBoxes(  # pyright: ignore[reportCallIssue]
            torch.tensor(boxes, dtype=torch.float32, device=device),
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=(image_size, image_size),
        ),
        labels=torch.randint(1, 80, (k,), generator=gen, dtype=torch.int64).to(device),
        instance_ids=torch.arange(k, dtype=torch.int32, device=device),
        instance_masks=InstanceMask(masks.to(device)),
    )


def build_batch(
    *,
    batch_size: int,
    image_size: int,
    k_range: tuple[int, int],
    max_instances: int,
    seed: int,
    device: torch.device,
) -> PaddedBatchedDenseSample:
    """Deterministic padded batch; ``k`` per sample drawn from ``k_range``."""
    rng = random.Random(seed)
    k_lo, k_hi = k_range
    samples = [
        _sample(image_size, rng.randint(k_lo, k_hi), seed * 1000 + i, device)
        for i in range(batch_size)
    ]
    return BatchedDenseSample.from_samples(samples).to_padded(
        max_instances=max_instances
    )


def build_batches(
    *,
    n_batches: int,
    batch_size: int,
    image_size: int,
    k_range: tuple[int, int],
    max_instances: int,
    device: torch.device,
    base_seed: int = 0,
) -> list[PaddedBatchedDenseSample]:
    return [
        build_batch(
            batch_size=batch_size,
            image_size=image_size,
            k_range=k_range,
            max_instances=max_instances,
            seed=base_seed + i,
            device=device,
        )
        for i in range(n_batches)
    ]
