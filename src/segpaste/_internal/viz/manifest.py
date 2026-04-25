"""Pydantic v2 models for the JSON artifacts emitted by the viz ritual.

The same models double as the schema enforced by the smoke test, so
adding a field here also tightens the test contract.
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict


class _FrozenStrict(BaseModel):
    """Local base: frozen, no extras (matches `BatchCopyPasteConfig`)."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class InvariantReportEntry(_FrozenStrict):
    """Wire-format mirror of `_internal.invariants.InvariantReport`.

    Mirrored rather than re-exported so the manifest schema is closed
    against changes to the in-process `InvariantReport` and so the JSON
    payload stays a stable cross-version artifact.
    """

    name: str
    ok: bool
    message: str | None = None
    details: Mapping[str, int | float | str] | None = None


class SampleInvariantLog(_FrozenStrict):
    """Per-sample invariant outcomes."""

    sample_index: int
    reports: tuple[InvariantReportEntry, ...]

    @property
    def ok(self) -> bool:
        return all(r.ok for r in self.reports)


class InvariantLog(_FrozenStrict):
    """Top-level container for `invariant_log.json`."""

    schema_version: int = 1
    preset: str
    seed: int
    samples: tuple[SampleInvariantLog, ...]


class DatasetSampleEntry(_FrozenStrict):
    """One row of `dataset_manifest.json`."""

    sample_index: int
    sha256: str
    height: int
    width: int


class DatasetManifest(_FrozenStrict):
    """Top-level container for `dataset_manifest.json`."""

    schema_version: int = 1
    source: str
    sample_count: int
    samples: tuple[DatasetSampleEntry, ...]


class RunManifest(_FrozenStrict):
    """Top-level container for `run_manifest.json` (ADR-0009 §5)."""

    schema_version: int = 1
    preset: str
    seed: int
    torch_version: str
    segpaste_version: str
    runner: str
    iso_date: str
    num_samples: int
    batch_size: int
    device: str
