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
from segpaste.augmentation.source import BankSource, IntraBatchSource, SourceStrategy


class IntraBatchSourceConfig(BaseModel):
    """Default source: pick a paste source from the same batch (v0.3.0)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["intra_batch"] = "intra_batch"


class BankSourceConfig(BaseModel):
    """External instance-bank source (ADR-0011 PR7).

    The bank itself (a backend opening on disk) is constructed
    imperatively and either wired through a ``DataLoader`` (preferred —
    see :func:`segpaste._internal.bank.loader.create_bank_dataloader`) or
    set per-step on the strategy via :meth:`BankSource.set_bank_batch`.
    The Pydantic config carries only the placement geometry knobs so
    YAML configs round-trip without serializing tensor state.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["bank"] = "bank"
    placement: BatchedPlacementConfig | None = None
    """Override for the parent ``BatchCopyPasteConfig.placement``. ``None``
    inherits the parent's placement config (typical)."""


SourceConfig = Annotated[
    IntraBatchSourceConfig | BankSourceConfig,
    Field(discriminator="kind"),
]
"""Discriminated union of source-selection strategies (ADR-0011)."""


def build_source_strategy(
    source: SourceConfig, placement: BatchedPlacementConfig
) -> SourceStrategy:
    """Materialize a :class:`SourceStrategy` from its config."""
    match source.kind:
        case "intra_batch":
            return IntraBatchSource(placement)
        case "bank":
            # Pyright narrows ``source`` to ``BankSourceConfig`` via the
            # discriminator literal — no isinstance call needed.
            return BankSource(source.placement or placement)
