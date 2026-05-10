"""Shared meta-file utilities for bank backends (ADR-0011)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FORMAT_VERSION = 1
EMBED_DIM = 256
EMB_BYTES = EMBED_DIM * 2  # float16


def meta_path(root: Path) -> Path:
    return root / "meta.json"


def hash_meta(meta: dict[str, Any]) -> str:
    """SHA-256 of the meta object with the ``sha256`` field excluded."""
    payload = {k: v for k, v in meta.items() if k != "sha256"}
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


@dataclass(frozen=True, slots=True)
class BankMeta:
    """Validated handle on a bank's ``meta.json``."""

    raw: dict[str, Any]
    n: int
    h: int
    w: int
    num_classes: int
    has_embeddings: bool
    sha256: str


def open_meta(root: Path, *, expected_format: str) -> BankMeta:
    """Load ``meta.json`` and validate its format, version, and sha256.

    Centralizes the validation block shared by every backend; per-backend
    ``__init__`` calls this and then opens its own data files.
    """
    with meta_path(root).open("r") as fh:
        raw: dict[str, Any] = json.load(fh)
    if raw.get("format") != expected_format:
        raise ValueError(
            f"{root}: meta format {raw.get('format')!r} != {expected_format!r}"
        )
    if raw.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"{root}: format_version {raw.get('format_version')} != {FORMAT_VERSION}"
        )
    recomputed = hash_meta(raw)
    stored = raw.get("sha256")
    if stored is not None and stored != recomputed:
        raise ValueError(
            f"{root}: meta.json sha256 mismatch (stored {stored!r}, "
            f"recomputed {recomputed!r})"
        )
    return BankMeta(
        raw=raw,
        n=int(raw["num_crops"]),
        h=int(raw["crop_h"]),
        w=int(raw["crop_w"]),
        num_classes=int(raw["num_classes"]),
        has_embeddings=bool(raw["has_embeddings"]),
        sha256=recomputed,
    )


def stamp_meta(root: Path, meta: dict[str, Any]) -> None:
    """Compute ``sha256`` over ``meta`` and write it to ``meta.json``."""
    meta["sha256"] = hash_meta(meta)
    with meta_path(root).open("w") as fh:
        json.dump(meta, fh, sort_keys=True, indent=2)


def base_meta(
    *,
    backend_format: str,
    n: int,
    h: int,
    w: int,
    num_classes: int,
    has_embeddings: bool,
    build_seed: int,
    segpaste_version: str,
    class_frequencies: list[int],
) -> dict[str, Any]:
    """Per-backend writers seed ``meta.json`` from this template."""
    return {
        "format": backend_format,
        "format_version": FORMAT_VERSION,
        "num_crops": int(n),
        "crop_h": int(h),
        "crop_w": int(w),
        "num_classes": int(num_classes),
        "has_embeddings": bool(has_embeddings),
        "segpaste_version": str(segpaste_version),
        "build_seed": int(build_seed),
        "class_frequencies": [int(x) for x in class_frequencies],
    }


__all__ = [
    "EMBED_DIM",
    "EMB_BYTES",
    "FORMAT_VERSION",
    "BankMeta",
    "base_meta",
    "hash_meta",
    "meta_path",
    "open_meta",
    "stamp_meta",
]
