"""torchvision reference ``SimpleCopyPaste`` adapter (ADR-0016 §1).

Wraps the vendored class at
:mod:`benchmarks.comparison._refs.torchvision_simple_copy_paste`. The
upstream class is reference training-script code; it consumes
``list[Tensor[C, H, W]]`` + ``list[dict[str, Tensor]]`` with the
canonical ``{masks, boxes, labels}`` keys.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from benchmarks.comparison._refs.torchvision_simple_copy_paste import SimpleCopyPaste
from benchmarks.comparison.impls._base import register
from benchmarks.comparison.workload import CanonicalSample

TorchvisionBatch = tuple[list[Tensor], list[dict[str, Tensor]]]


class TorchvisionRefImpl:
    """Adapter around torchvision's reference :class:`SimpleCopyPaste`."""

    name = "torchvision_ref"

    def __init__(self) -> None:
        self._module: SimpleCopyPaste | None = None

    def supports_device(self, device: torch.device) -> bool:
        del device
        return True

    def adapt(
        self, batches: Sequence[Sequence[CanonicalSample]]
    ) -> list[TorchvisionBatch]:
        self._module = SimpleCopyPaste(blending=True)
        return [_pack(batch) for batch in batches]

    def step(self, batch: TorchvisionBatch) -> object:
        if self._module is None:
            raise RuntimeError("call adapt() before step()")
        images, targets = batch
        return self._module(images, targets)


def _pack(batch: Sequence[CanonicalSample]) -> TorchvisionBatch:
    images: list[Tensor] = []
    targets: list[dict[str, Tensor]] = []
    for s in batch:
        images.append(s.image)
        targets.append(
            {
                "masks": s.masks,
                "boxes": s.boxes,
                "labels": s.labels,
            }
        )
    return images, targets


register("torchvision_ref", TorchvisionRefImpl)
