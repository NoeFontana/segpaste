"""Schemas for the preset registry (ADR-0009 §3, §5).

Both models are frozen Pydantic v2 with ``extra="forbid"`` (precedent:
:class:`BatchCopyPasteConfig`, ADR-0008).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from segpaste.augmentation.batch_copy_paste import BatchCopyPasteConfig
from segpaste.types import Modality


class _FrozenStrict(BaseModel):
    """Local base for the two preset schemas: frozen, no extras."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class SignOff(_FrozenStrict):
    """Audit-trail metadata for the local validation ritual (ADR-0009 §5)."""

    torch_version: str
    """Output of ``torch.__version__`` at the time of the local run."""

    seed: int = 0xC0FFEE
    """``Generator().manual_seed(seed)`` used for the local run."""

    sample_count: int
    """Number of dataset samples the visualizer iterated over."""

    iso_date: str
    """ISO-8601 date string of the local run (UTC)."""


class PresetConfig(_FrozenStrict):
    """A registered dataset preset (ADR-0009 §3).

    Field additions are allowed (additive-only per ADR-0001 Part (i));
    renames or removals are breaking.
    """

    name: str
    """Stable identifier; matches the registry key."""

    description: str
    """One-paragraph human-readable rationale."""

    batch_copy_paste: BatchCopyPasteConfig = Field(default_factory=BatchCopyPasteConfig)
    """The augmentation hyperparameters this preset pins."""

    target_modalities: tuple[Modality, ...]
    """Dense-sample modalities this preset expects to see."""

    sign_off: SignOff | None = None
    """Audit trail for the local sign-off ritual (ADR-0009 §5)."""
