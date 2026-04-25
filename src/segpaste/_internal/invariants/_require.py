"""Shared helper for unwrapping modality-gated fields."""

from typing import TypeVar

_T = TypeVar("_T")


def require(value: _T | None, msg: str) -> _T:
    """Unwrap a modality-gated field, raising ``AssertionError`` if absent."""
    if value is None:
        raise AssertionError(msg)
    return value
