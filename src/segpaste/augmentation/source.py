"""Source-selection strategies for :class:`BatchCopyPaste` (ADR-0011).

Abstracts over *where the paste sources come from* so :class:`BatchCopyPaste`
can operate either on intra-batch sources (the v0.3.0 default) or on an
external instance bank without changing its forward graph.

The protocol returns a ``(source_view, placement)`` tuple where ``source_view``
is a :class:`PaddedBatchedDenseSample` row-aligned with ``target`` (matching
``B``) and ``placement.source_idx`` indexes into the source view (not the
target). For :class:`IntraBatchSource`, ``source_view is target`` and
``source_idx`` is drawn off-diagonal — bitwise identical to v0.3.0.

The forward graph in :class:`BatchCopyPaste` calls
``self.source_strategy.sample(...)`` once and feeds the result to
:class:`AffinePropagator`. No graph branching on the strategy type — both
strategies return the same shapes.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from torch import Tensor, nn

from segpaste._internal.gpu.batched_placement import (
    BatchedPlacement,
    BatchedPlacementConfig,
    BatchedPlacementSampler,
)
from segpaste.types import PaddedBatchedDenseSample


@runtime_checkable
class SourceStrategy(Protocol):
    """Picks the source view and per-target placement for one forward step.

    Implementations may be ``nn.Module`` subclasses (so child modules and
    buffers register correctly) or plain callables — :func:`runtime_checkable`
    structural typing only requires ``sample``. The return contract is fixed:
    ``source_view`` row-aligned with ``target`` along the batch dim, and
    ``placement.source_idx`` indexing into ``source_view``.
    """

    def sample(
        self,
        target: PaddedBatchedDenseSample,
        valid_extent: Tensor | None,
        source_eligible: Tensor | None,
        generator: torch.Generator | None,
    ) -> tuple[PaddedBatchedDenseSample, BatchedPlacement]: ...


class IntraBatchSource(nn.Module):
    """v0.3.0-equivalent source: sample sources from the same batch.

    Wraps :class:`BatchedPlacementSampler` and returns ``target`` itself as
    the source view. The diagonal-masked multinomial inside the sampler
    guarantees ``source_idx[i] != i`` for ``B > 1``. Default constructor
    matches v0.3.0 defaults; pass a :class:`BatchedPlacementConfig` for
    non-default placement parameters.
    """

    def __init__(self, config: BatchedPlacementConfig | None = None) -> None:
        super().__init__()
        self.placement_sampler = BatchedPlacementSampler(config)

    def sample(
        self,
        target: PaddedBatchedDenseSample,
        valid_extent: Tensor | None,
        source_eligible: Tensor | None,
        generator: torch.Generator | None,
    ) -> tuple[PaddedBatchedDenseSample, BatchedPlacement]:
        placement = self.placement_sampler(
            target,
            generator,
            valid_extent=valid_extent,
            source_eligible=source_eligible,
        )
        return target, placement
