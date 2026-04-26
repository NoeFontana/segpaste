"""Registry behavior for `segpaste.presets` (ADR-0009)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from pydantic import ValidationError

from segpaste.presets import (
    PresetConfig,
    SignOff,
    get_preset,
    list_presets,
    register_preset,
)
from segpaste.types import Modality


@pytest.fixture(autouse=True)
def _isolate_registry() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    """Restore the module-level registry after each test.

    Built-in presets are autoloaded by ``segpaste.presets`` at import time;
    snapshot, clear, run, and restore so each test sees a clean registry.
    """
    from segpaste import presets

    registry = presets._REGISTRY  # pyright: ignore[reportPrivateUsage]
    saved = dict(registry)
    registry.clear()
    yield
    registry.clear()
    registry.update(saved)


def _make(name: str = "synthetic-smoke") -> PresetConfig:
    return PresetConfig(
        name=name,
        description="Synthetic smoke preset for registry tests.",
        target_modalities=(Modality.IMAGE, Modality.INSTANCE),
    )


def test_registry_starts_empty() -> None:
    assert list_presets() == ()


def test_register_then_get_roundtrip() -> None:
    cfg = _make()
    register_preset(cfg)
    assert get_preset(cfg.name) is cfg


def test_list_presets_is_sorted_value_snapshot() -> None:
    register_preset(_make("zeta"))
    register_preset(_make("alpha"))
    names = list_presets()
    assert names == ("alpha", "zeta")
    register_preset(_make("mu"))
    assert names == ("alpha", "zeta")
    assert list_presets() == ("alpha", "mu", "zeta")


def test_duplicate_registration_raises() -> None:
    register_preset(_make("dup"))
    with pytest.raises(ValueError, match="already registered"):
        register_preset(_make("dup"))


def test_unknown_preset_raises_keyerror() -> None:
    with pytest.raises(KeyError, match="unknown preset"):
        get_preset("does-not-exist")


def test_preset_config_is_frozen() -> None:
    cfg = _make()
    with pytest.raises(ValidationError):
        cfg.name = "mutated"  # type: ignore[misc]


def test_preset_config_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        PresetConfig(  # pyright: ignore[reportCallIssue]
            name="x",
            description="x",
            target_modalities=(Modality.IMAGE,),
            unexpected_field=1,  # pyright: ignore[reportCallIssue]
        )


def test_preset_config_is_hashable() -> None:
    cfg_a = _make("hash-a")
    cfg_b = _make("hash-a")
    assert hash(cfg_a) == hash(cfg_b)
    cfg_c = _make("hash-b")
    assert hash(cfg_a) != hash(cfg_c)


def test_signoff_round_trip() -> None:
    sign_off = SignOff(
        torch_version="2.8.0",
        sample_count=128,
        iso_date="2026-04-25",
    )
    cfg = PresetConfig(
        name="with-sign-off",
        description="d",
        target_modalities=(Modality.IMAGE,),
        sign_off=sign_off,
    )
    assert cfg.sign_off == sign_off
    assert sign_off.seed == 0xC0FFEE
