"""Manifest-driven loader for golden fixtures."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import torch

from segpaste.types import DenseSample, Modality
from tests.fixtures.synthetic import BUILDERS

_MANIFEST_PATH = Path(__file__).parent / "manifest.toml"
_SYNTHETIC_DIR = Path(__file__).parent / "synthetic"


@dataclass(frozen=True, slots=True)
class FixtureRecord:
    name: str
    modalities: frozenset[Modality]
    shape: tuple[int, ...]
    source: str
    licence: str
    builder: str


def _parse_modalities(names: list[str]) -> frozenset[Modality]:
    return frozenset(Modality(n) for n in names)


def _load_manifest() -> dict[str, FixtureRecord]:
    with _MANIFEST_PATH.open("rb") as fh:
        raw = tomllib.load(fh)
    out: dict[str, FixtureRecord] = {}
    for entry in raw.get("fixture", []):
        out[entry["name"]] = FixtureRecord(
            name=entry["name"],
            modalities=_parse_modalities(entry["modalities"]),
            shape=tuple(entry["shape"]),
            source=entry["source"],
            licence=entry["licence"],
            builder=entry["builder"],
        )
    return out


FIXTURE_MANIFEST: dict[str, FixtureRecord] = _load_manifest()


def fixture_names(
    requires: frozenset[Modality] | set[Modality] | None = None,
) -> list[str]:
    """Return the names of fixtures whose modality set ⊇ ``requires``."""
    if requires is None:
        return sorted(FIXTURE_MANIFEST)
    required = frozenset(requires)
    return sorted(
        name
        for name, rec in FIXTURE_MANIFEST.items()
        if required.issubset(rec.modalities)
    )


@cache
def load_fixture(name: str) -> DenseSample:
    """Load ``name`` from its committed ``.pt`` if present, else build on the fly."""
    if name not in FIXTURE_MANIFEST:
        raise KeyError(f"unknown fixture: {name!r}")
    pt_path = _SYNTHETIC_DIR / f"{name}.pt"
    if pt_path.is_file():
        data = torch.load(pt_path, weights_only=False)
        return DenseSample.from_dict(data)
    builder = BUILDERS.get(name)
    if builder is None:
        raise FileNotFoundError(
            f"fixture {name!r}: no committed .pt and no registered builder"
        )
    return builder()


def regenerate_all() -> None:
    """Rebuild every committed ``.pt`` from its deterministic builder.

    Entry point for ``python -m tests.fixtures.loader`` — regenerates the
    committed blobs after a builder change.
    """
    _SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    for name, builder in BUILDERS.items():
        sample = builder()
        torch.save(sample.to_dict(), _SYNTHETIC_DIR / f"{name}.pt")
    print(f"regenerated {len(BUILDERS)} fixture(s) in {_SYNTHETIC_DIR}")


if __name__ == "__main__":
    regenerate_all()
