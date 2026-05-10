"""Smoke + determinism tests for :func:`build_bank` (ADR-0011 PR5)."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pytest
import torch

np = pytest.importorskip("numpy")

from segpaste._internal.bank.build import build_bank  # noqa: E402
from segpaste._internal.bank.memmap import MemmapBank  # noqa: E402
from segpaste._internal.bank.protocol import BankCrop  # noqa: E402


def _synthetic_crops(
    *, n: int, h: int, w: int, seed: int, with_embeddings: bool = False
) -> Iterable[BankCrop]:
    rng = np.random.default_rng(seed)
    for i in range(n):
        image = rng.integers(0, 256, size=(3, h, w), dtype=np.uint8)
        alpha = rng.integers(0, 2, size=(1, h, w)).astype(np.bool_)
        embedding = (
            rng.standard_normal((256,)).astype(np.float16) if with_embeddings else None
        )
        yield BankCrop(
            image=image,  # type: ignore[arg-type]
            alpha=alpha,  # type: ignore[arg-type]
            class_id=int(i % 4),
            embedding=embedding,  # type: ignore[arg-type]
        )


def test_build_memmap_bank_roundtrips(tmp_path: Path) -> None:
    crops = _synthetic_crops(n=6, h=8, w=8, seed=0)
    out = build_bank(
        crops,
        out_path=tmp_path,
        out_format="memmap",
        num_classes=4,
        crop_size=(8, 8),
    )
    bank = MemmapBank(out)
    assert len(bank) == 6
    assert bank.crop_size == (8, 8)


def test_build_two_runs_byte_identical(tmp_path: Path) -> None:
    out_a = build_bank(
        _synthetic_crops(n=4, h=8, w=8, seed=42),
        out_path=tmp_path / "a",
        out_format="memmap",
        num_classes=4,
        crop_size=(8, 8),
    )
    out_b = build_bank(
        _synthetic_crops(n=4, h=8, w=8, seed=42),
        out_path=tmp_path / "b",
        out_format="memmap",
        num_classes=4,
        crop_size=(8, 8),
    )
    assert MemmapBank(out_a).version == MemmapBank(out_b).version


def test_build_lmdb_smoke(tmp_path: Path) -> None:
    pytest.importorskip("lmdb")
    from segpaste._internal.bank.lmdb_backend import LMDBBank

    out = build_bank(
        _synthetic_crops(n=4, h=8, w=8, seed=0),
        out_path=tmp_path,
        out_format="lmdb",
        num_classes=4,
        crop_size=(8, 8),
    )
    bank = LMDBBank(out)
    assert len(bank) == 4


def test_build_with_embeddings_round_trip(tmp_path: Path) -> None:
    out = build_bank(
        _synthetic_crops(n=3, h=8, w=8, seed=0, with_embeddings=True),
        out_path=tmp_path,
        out_format="memmap",
        num_classes=4,
        crop_size=(8, 8),
    )
    bank = MemmapBank(out)
    assert bank.has_embeddings
    for i in range(len(bank)):
        emb = bank[i].embedding
        assert emb is not None
        assert emb.shape == (256,)


def test_inconsistent_embedding_presence_rejected(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)

    def crops() -> Iterable[BankCrop]:
        yield BankCrop(
            image=rng.integers(0, 256, (3, 8, 8), dtype=np.uint8),  # type: ignore[arg-type]
            alpha=rng.integers(0, 2, (1, 8, 8)).astype(np.bool_),  # type: ignore[arg-type]
            class_id=0,
            embedding=torch.zeros((256,), dtype=torch.float16),
        )
        yield BankCrop(
            image=rng.integers(0, 256, (3, 8, 8), dtype=np.uint8),  # type: ignore[arg-type]
            alpha=rng.integers(0, 2, (1, 8, 8)).astype(np.bool_),  # type: ignore[arg-type]
            class_id=1,
            embedding=None,
        )

    with pytest.raises(ValueError, match="inconsistent embedding"):
        build_bank(
            crops(),
            out_path=tmp_path,
            out_format="memmap",
            num_classes=4,
            crop_size=(8, 8),
        )


def test_unknown_format_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown out_format"):
        build_bank(
            _synthetic_crops(n=2, h=8, w=8, seed=0),
            out_path=tmp_path,
            out_format="hdf5",  # type: ignore[arg-type]
            num_classes=4,
            crop_size=(8, 8),
        )


def test_empty_iterable_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty iterable"):
        build_bank(
            iter([]),
            out_path=tmp_path,
            out_format="memmap",
            num_classes=4,
            crop_size=(8, 8),
        )
