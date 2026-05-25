"""segpaste :class:`BatchCopyPaste` adapter for the comparison harness."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torchvision import tv_tensors

from benchmarks.comparison.impls._base import register
from benchmarks.comparison.workload import CanonicalSample
from segpaste import BatchCopyPaste, PaddedBatchedDenseSample
from segpaste.types import BatchedDenseSample, DenseSample, InstanceMask


class SegpasteImpl:
    """Wraps :class:`segpaste.BatchCopyPaste`."""

    name = "segpaste"

    def __init__(self) -> None:
        self._module: BatchCopyPaste | None = None
        self._generator: torch.Generator | None = None

    def supports_device(self, device: torch.device) -> bool:
        del device
        return True

    def adapt(
        self, batches: Sequence[Sequence[CanonicalSample]]
    ) -> list[PaddedBatchedDenseSample]:
        if not batches:
            return []
        first_batch = batches[0]
        if not first_batch:
            return []
        device = first_batch[0].image.device
        max_instances = max(s.masks.shape[0] for batch in batches for s in batch)

        self._module = BatchCopyPaste().to(device)
        self._generator = torch.Generator(device="cpu").manual_seed(0)

        return [self._pack(batch, max_instances) for batch in batches]

    def step(self, batch: PaddedBatchedDenseSample) -> object:
        if self._module is None or self._generator is None:
            raise RuntimeError("call adapt() before step()")
        return self._module(batch, self._generator)

    @staticmethod
    def _pack(
        batch: Sequence[CanonicalSample], max_instances: int
    ) -> PaddedBatchedDenseSample:
        samples = [
            DenseSample(
                image=tv_tensors.Image(s.image),
                boxes=tv_tensors.BoundingBoxes(  # pyright: ignore[reportCallIssue]
                    s.boxes,
                    format=tv_tensors.BoundingBoxFormat.XYXY,
                    canvas_size=(int(s.image.shape[-2]), int(s.image.shape[-1])),
                ),
                labels=s.labels,
                instance_ids=torch.arange(
                    s.masks.shape[0], dtype=torch.int32, device=s.image.device
                ),
                instance_masks=InstanceMask(s.masks),
            )
            for s in batch
        ]
        return BatchedDenseSample.from_samples(samples).to_padded(
            max_instances=max_instances
        )


register("segpaste", SegpasteImpl)
