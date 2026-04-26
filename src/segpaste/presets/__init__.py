"""Dataset preset registry (ADR-0009).

Public surface: :func:`register_preset`, :func:`get_preset`,
:func:`list_presets`, :class:`PresetConfig`, :class:`SignOff`.
"""

from __future__ import annotations

from segpaste.presets._base import PresetConfig, SignOff

__all__ = [
    "PresetConfig",
    "SignOff",
    "get_preset",
    "list_presets",
    "register_preset",
]

_REGISTRY: dict[str, PresetConfig] = {}


def register_preset(config: PresetConfig) -> None:
    """Register *config* under ``config.name``.

    Raises:
        ValueError: if ``config.name`` is already registered.
    """
    if config.name in _REGISTRY:
        raise ValueError(f"preset {config.name!r} is already registered")
    _REGISTRY[config.name] = config


def get_preset(name: str) -> PresetConfig:
    """Return the registered preset for *name*.

    Raises:
        KeyError: if *name* is not registered.
    """
    if name not in _REGISTRY:
        raise KeyError(f"unknown preset {name!r}; registered: {list_presets()}")
    return _REGISTRY[name]


def list_presets() -> tuple[str, ...]:
    """Sorted tuple of registered preset names. The result is a value snapshot."""
    return tuple(sorted(_REGISTRY))


# Side-effect imports register built-in presets — must follow the helpers above.
import segpaste.presets.coco_instance  # noqa: E402  # pyright: ignore[reportUnusedImport]
import segpaste.presets.coco_panoptic  # noqa: E402, F401  # pyright: ignore[reportUnusedImport]
