"""Canonical workload shared across implementations (ADR-0016 §2).

A :class:`Workload` is the single source of per-sample data. Each
implementation's :meth:`Implementation.adapt` consumes the same
``list[list[CanonicalSample]]`` and converts it into the impl's native
input shape. The timer (`benchmarks/_harness.py`) measures
:meth:`Implementation.step` only; adapter cost is outside the window.

Sample shape is the lowest common denominator that all three reference
implementations (segpaste, torchvision-ref, mmdet) can consume:
``image [3, H, W] float32``, ``masks [N, H, W] bool``,
``boxes [N, 4] float32 xyxy``, ``labels [N] int64``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch


@dataclass(frozen=True, slots=True)
class CanonicalSample:
    """Per-sample input shared by every implementation in the comparison."""

    image: torch.Tensor  # [3, H, W] float32 in [0, 1]
    masks: torch.Tensor  # [N, H, W] bool
    boxes: torch.Tensor  # [N, 4] float32 xyxy
    labels: torch.Tensor  # [N] int64


@dataclass(frozen=True, slots=True)
class Workload:
    """Deterministic workload spec for the comparison sweep (ADR-0016 §3)."""

    batch_size: int
    image_size: int
    k_range: tuple[int, int]
    max_instances: int
    seed: int = 0
    n_batches: int = 8
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    def label(self) -> str:
        """Short identifier for log lines and JSON sub_label."""
        return (
            f"b{self.batch_size}_img{self.image_size}"
            f"_k{self.k_range[0]}-{self.k_range[1]}_{self.device.type}"
        )

    def build_batches(self) -> list[list[CanonicalSample]]:
        """Pre-build ``n_batches`` independent batches.

        Each batch holds ``batch_size`` :class:`CanonicalSample` instances.
        Reproducible: identical ``(seed, n_batches, batch_size, image_size,
        k_range)`` always produces the same tensors.
        """
        rng = random.Random(self.seed)
        return [
            [
                _build_sample(
                    image_size=self.image_size,
                    k=rng.randint(self.k_range[0], self.k_range[1]),
                    seed=self.seed * 1_000_003 + batch_idx * 1_009 + sample_idx,
                    device=self.device,
                )
                for sample_idx in range(self.batch_size)
            ]
            for batch_idx in range(self.n_batches)
        ]


def _build_sample(
    *, image_size: int, k: int, seed: int, device: torch.device
) -> CanonicalSample:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    image = torch.rand(3, image_size, image_size, generator=gen, dtype=torch.float32)
    masks = torch.zeros(k, image_size, image_size, dtype=torch.bool)
    raw_boxes: list[list[float]] = []
    side_min = min(8, max(1, image_size // 8))
    side_max = max(side_min + 1, image_size // 4)
    for i in range(k):
        side = int(torch.randint(side_min, side_max, (1,), generator=gen).item())
        x1 = int(torch.randint(0, image_size - side, (1,), generator=gen).item())
        y1 = int(torch.randint(0, image_size - side, (1,), generator=gen).item())
        masks[i, y1 : y1 + side, x1 : x1 + side] = True
        raw_boxes.append([float(x1), float(y1), float(x1 + side), float(y1 + side)])
    labels = torch.randint(1, 80, (k,), generator=gen, dtype=torch.int64)
    boxes = torch.tensor(raw_boxes, dtype=torch.float32)
    return CanonicalSample(
        image=image.to(device),
        masks=masks.to(device),
        boxes=boxes.to(device),
        labels=labels.to(device),
    )
