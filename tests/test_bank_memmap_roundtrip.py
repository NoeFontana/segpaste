"""Round-trip tests for :class:`MemmapBank` (ADR-0011 PR4)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

np = pytest.importorskip("numpy")

from segpaste import InstanceBank  # noqa: E402
from segpaste._internal.bank.memmap import (  # noqa: E402
    MemmapBank,
    write_memmap_bank,
)


def _build_bank(
    tmp_path: Path,
    *,
    n: int = 16,
    h: int = 12,
    w: int = 10,
    num_classes: int = 5,
    with_embeddings: bool = False,
    seed: int = 0,
) -> tuple[Path, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    images = rng.integers(0, 256, size=(n, 3, h, w), dtype=np.uint8)
    alpha = rng.integers(0, 2, size=(n, 1, h, w)).astype(np.bool_)
    classes = rng.integers(0, num_classes, size=(n,), dtype=np.int64)
    embeddings = None
    if with_embeddings:
        embeddings = rng.standard_normal((n, 256)).astype(np.float16)
    write_memmap_bank(
        tmp_path,
        images=images,
        alpha=alpha,
        classes=classes,
        num_classes=num_classes,
        embeddings=embeddings,
        build_seed=seed,
        segpaste_version="0.4.0-dev",
    )
    return tmp_path, {
        "images": images,
        "alpha": alpha,
        "classes": classes,
        "embeddings": embeddings,
    }


def test_memmap_bank_satisfies_protocol(tmp_path: Path) -> None:
    root, _ = _build_bank(tmp_path)
    bank: InstanceBank = MemmapBank(root)
    assert isinstance(bank, InstanceBank)


def test_round_trip_byte_identical(tmp_path: Path) -> None:
    root, ref = _build_bank(tmp_path, n=8)
    bank = MemmapBank(root)
    assert len(bank) == 8
    for i in range(len(bank)):
        crop = bank[i]
        assert torch.equal(crop.image, torch.from_numpy(ref["images"][i].copy()))
        assert torch.equal(crop.alpha, torch.from_numpy(ref["alpha"][i].copy()))
        assert crop.class_id == int(ref["classes"][i])
        assert crop.embedding is None


def test_embeddings_round_trip(tmp_path: Path) -> None:
    root, ref = _build_bank(tmp_path, n=4, with_embeddings=True)
    bank = MemmapBank(root)
    assert bank.has_embeddings
    for i in range(len(bank)):
        emb = bank[i].embedding
        assert emb is not None
        assert torch.equal(emb, torch.from_numpy(ref["embeddings"][i].copy()))


def test_class_frequencies_match_classes(tmp_path: Path) -> None:
    root, ref = _build_bank(tmp_path, n=24, num_classes=6)
    bank = MemmapBank(root)
    expected = torch.bincount(torch.from_numpy(ref["classes"]), minlength=6)
    assert torch.equal(bank.class_frequencies, expected)


def test_crop_class_ids_round_trip(tmp_path: Path) -> None:
    root, ref = _build_bank(tmp_path, n=8)
    bank = MemmapBank(root)
    expected = torch.from_numpy(ref["classes"].copy())
    assert torch.equal(bank.crop_class_ids, expected)


def test_version_string_format(tmp_path: Path) -> None:
    root, _ = _build_bank(tmp_path)
    bank = MemmapBank(root)
    assert bank.version.startswith("memmap@")
    assert len(bank.version) == len("memmap@") + 12


def test_meta_sha256_drift_rejected(tmp_path: Path) -> None:
    root, _ = _build_bank(tmp_path)
    meta_path = root / "meta.json"
    with meta_path.open("r") as fh:
        meta = json.load(fh)
    meta["build_seed"] = meta["build_seed"] + 1  # silently mutate without re-hashing
    with meta_path.open("w") as fh:
        json.dump(meta, fh, sort_keys=True, indent=2)
    with pytest.raises(ValueError, match="sha256 mismatch"):
        MemmapBank(root)


def test_two_writes_with_same_inputs_are_byte_identical(tmp_path: Path) -> None:
    a, _ = _build_bank(tmp_path / "a", seed=42)
    b, _ = _build_bank(tmp_path / "b", seed=42)
    bank_a = MemmapBank(a)
    bank_b = MemmapBank(b)
    assert bank_a.version == bank_b.version


def test_index_out_of_range_raises(tmp_path: Path) -> None:
    root, _ = _build_bank(tmp_path, n=3)
    bank = MemmapBank(root)
    with pytest.raises(IndexError):
        bank[5]
