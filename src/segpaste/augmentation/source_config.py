"""Pydantic config layer for :class:`SourceStrategy` (ADR-0011).

Discriminated union over source-selection strategies. Only
:class:`IntraBatchSourceConfig` exists at A1 PR3; A1 PR7 lands
``BankSourceConfig`` for the external instance bank. Discriminator field
is ``kind`` so YAML configs round-trip cleanly under
``model_validate``/``model_dump``.

The :func:`build_source_strategy` factory turns the config into a live
:class:`SourceStrategy` instance, threading in the
:class:`BatchedPlacementConfig` carried by
:class:`segpaste.augmentation.batch_copy_paste.BatchCopyPasteConfig`.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from segpaste._internal.gpu.batched_placement import BatchedPlacementConfig
from segpaste.augmentation.source import IntraBatchSource, SourceStrategy


class IntraBatchSourceConfig(BaseModel):
    """Default source: pick a paste source from the same batch (v0.3.0)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["intra_batch"] = "intra_batch"


SourceConfig = Annotated[
    IntraBatchSourceConfig,
    Field(discriminator="kind"),
]
"""Discriminated union of source strategies. A1 PR7 widens this to include
``BankSourceConfig``; until then the union is degenerate (one variant)."""


def build_source_strategy(
    source: SourceConfig, placement: BatchedPlacementConfig
) -> SourceStrategy:
    """Materialize a :class:`SourceStrategy` from its config.

    ``placement`` is threaded through from the parent
    :class:`BatchCopyPasteConfig` because :class:`IntraBatchSource` shares
    its placement parameters with the broader augmentation pipeline; the
    bank-source path (PR7) will derive its own placement params from
    ``BankSourceConfig``.
    """
    # PR3 ships a single-variant discriminated union; PR7 widens this
    # dispatch when ``BankSourceConfig`` lands.
    match source.kind:
        case "intra_batch":
            return IntraBatchSource(placement)
