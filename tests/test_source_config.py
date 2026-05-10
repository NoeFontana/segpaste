"""Round-trip and dispatch tests for :class:`SourceConfig` (ADR-0011 PR3)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from segpaste._internal.gpu.batched_placement import BatchedPlacementConfig
from segpaste.augmentation.batch_copy_paste import BatchCopyPasteConfig
from segpaste.augmentation.source import IntraBatchSource
from segpaste.augmentation.source_config import (
    IntraBatchSourceConfig,
    build_source_strategy,
)


def test_default_batch_copy_paste_config_uses_intra_batch_source() -> None:
    config = BatchCopyPasteConfig()
    assert isinstance(config.source, IntraBatchSourceConfig)
    assert config.source.kind == "intra_batch"


def test_dump_round_trip_via_dict() -> None:
    config = BatchCopyPasteConfig()
    raw = config.model_dump()
    assert raw["source"] == {"kind": "intra_batch"}
    rebuilt = BatchCopyPasteConfig.model_validate(raw)
    assert rebuilt == config


def test_dump_round_trip_via_json() -> None:
    config = BatchCopyPasteConfig()
    rebuilt = BatchCopyPasteConfig.model_validate_json(config.model_dump_json())
    assert rebuilt == config


def test_round_trip_omitting_source_field() -> None:
    """A YAML/JSON config without a ``source`` key still loads as v0.3.0 default."""
    rebuilt = BatchCopyPasteConfig.model_validate({})
    assert isinstance(rebuilt.source, IntraBatchSourceConfig)
    assert rebuilt.source.kind == "intra_batch"


def test_unknown_kind_rejected_by_discriminator() -> None:
    with pytest.raises(ValidationError):
        BatchCopyPasteConfig.model_validate({"source": {"kind": "totally-invalid"}})


def test_extra_field_on_source_rejected() -> None:
    with pytest.raises(ValidationError):
        BatchCopyPasteConfig.model_validate(
            {"source": {"kind": "intra_batch", "secret": 42}}
        )


def test_build_source_strategy_returns_intra_batch_source() -> None:
    placement = BatchedPlacementConfig()
    strategy = build_source_strategy(IntraBatchSourceConfig(), placement)
    assert isinstance(strategy, IntraBatchSource)
