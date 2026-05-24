"""Soft-dependency gateways: import-or-raise helpers for optional extras."""

from __future__ import annotations

from types import ModuleType


def require_fiftyone() -> ModuleType:
    """Return the imported ``fiftyone`` module or raise with an install hint."""
    try:
        import fiftyone
    except ImportError as exc:
        raise ImportError(
            "fiftyone is not installed. Install with `uv sync --group visualize`."
        ) from exc
    return fiftyone


def require_huggingface_hub() -> ModuleType:
    """Return the imported ``huggingface_hub`` module or raise with an install hint."""
    try:
        import huggingface_hub
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is not installed. Install with `uv sync --group eval`."
        ) from exc
    return huggingface_hub


def require_numpy() -> ModuleType:
    """Return the imported ``numpy`` module or raise with an install hint."""
    try:
        import numpy
    except ImportError as exc:
        raise ImportError(
            "numpy is not installed. Install with `uv sync --group bank-memmap`."
        ) from exc
    return numpy


def require_lmdb() -> ModuleType:
    """Return the imported ``lmdb`` module or raise with an install hint."""
    try:
        import lmdb
    except ImportError as exc:
        raise ImportError(
            "lmdb is not installed. Install with `uv sync --group bank-lmdb`."
        ) from exc
    return lmdb


def require_pyarrow() -> ModuleType:
    """Return ``pyarrow`` or raise with an install hint."""
    try:
        import pyarrow
    except ImportError as exc:
        raise ImportError(
            "pyarrow is not installed. Install with `uv sync --group bank-webdataset`."
        ) from exc
    return pyarrow


def require_lightning() -> ModuleType:
    """Return ``lightning.pytorch`` or raise with an install hint."""
    try:
        # lightning is an optional extra; absent in the base wheel.
        import lightning.pytorch  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise ImportError(
            "lightning is not installed. "
            "Install with `pip install 'segpaste[lightning]'`."
        ) from exc
    return lightning.pytorch
