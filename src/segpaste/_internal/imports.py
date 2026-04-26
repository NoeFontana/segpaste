"""Soft-dependency gateways: import-or-raise helpers for optional extras."""

from __future__ import annotations

from types import ModuleType


def require_fiftyone() -> ModuleType:
    """Return the imported ``fiftyone`` module or raise with an install hint."""
    try:
        import fiftyone
    except ImportError as exc:
        raise ImportError(
            "fiftyone is not installed. Install with `uv sync --group viewer`."
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
