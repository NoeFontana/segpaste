"""Round-trip tests for :class:`WebDatasetBank` (ADR-0011 PR6)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

np = pytest.importorskip("numpy")
pytest.importorskip("pyarrow")

from segpaste import InstanceBank  # noqa: E402
from segpaste._internal.bank.webdataset_backend import (  # noqa: E402
    WebDatasetBank,
    write_webdataset_bank,
)


def _build_bank(
    tmp_path: Path,
    *,
    n: int = 6,
    h: int = 8,
    w: int = 6,
    num_classes: int = 4,
    with_embeddings: bool = False,
    seed: int = 0,
    crops_per_shard: int = 3,
) -> tuple[Path, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    images = rng.integers(0, 256, size=(n, 3, h, w), dtype=np.uint8)
    alpha = rng.integers(0, 2, size=(n, 1, h, w)).astype(np.bool_)
    classes = rng.integers(0, num_classes, size=(n,), dtype=np.int64)
    embeddings = None
    if with_embeddings:
        embeddings = rng.standard_normal((n, 256)).astype(np.float16)
    write_webdataset_bank(
        tmp_path,
        images=images,
        alpha=alpha,
        classes=classes,
        num_classes=num_classes,
        embeddings=embeddings,
        build_seed=seed,
        segpaste_version="0.4.0-dev",
        crops_per_shard=crops_per_shard,
    )
    return tmp_path, {
        "images": images,
        "alpha": alpha,
        "classes": classes,
        "embeddings": embeddings,
    }


def test_webdataset_bank_satisfies_protocol(tmp_path: Path) -> None:
    root, _ = _build_bank(tmp_path)
    bank: InstanceBank = WebDatasetBank(root)
    assert isinstance(bank, InstanceBank)


def test_round_trip_byte_identical(tmp_path: Path) -> None:
    root, ref = _build_bank(tmp_path, n=5)
    bank = WebDatasetBank(root)
    assert len(bank) == 5
    for i in range(len(bank)):
        crop = bank[i]
        assert torch.equal(crop.image, torch.from_numpy(ref["images"][i].copy()))
        assert torch.equal(crop.alpha, torch.from_numpy(ref["alpha"][i].copy()))
        assert crop.class_id == int(ref["classes"][i])


def test_embeddings_round_trip(tmp_path: Path) -> None:
    root, ref = _build_bank(tmp_path, n=4, with_embeddings=True)
    bank = WebDatasetBank(root)
    assert bank.has_embeddings
    for i in range(len(bank)):
        emb = bank[i].embedding
        assert emb is not None
        assert torch.equal(emb, torch.from_numpy(ref["embeddings"][i].copy()))


def test_class_frequencies_match_classes(tmp_path: Path) -> None:
    root, ref = _build_bank(tmp_path, n=10, num_classes=5)
    bank = WebDatasetBank(root)
    expected = torch.bincount(torch.from_numpy(ref["classes"]), minlength=5)
    assert torch.equal(bank.class_frequencies, expected)


def test_two_writes_byte_identical(tmp_path: Path) -> None:
    a, _ = _build_bank(tmp_path / "a", seed=11)
    b, _ = _build_bank(tmp_path / "b", seed=11)
    assert WebDatasetBank(a).version == WebDatasetBank(b).version


def test_multi_shard_indexing(tmp_path: Path) -> None:
    # 7 crops across 3-crop shards = 3 shards (3, 3, 1 crops).
    root, ref = _build_bank(tmp_path, n=7, crops_per_shard=3)
    bank = WebDatasetBank(root)
    assert len(bank) == 7
    for i in range(7):
        assert bank[i].class_id == int(ref["classes"][i])


def test_index_out_of_range_raises(tmp_path: Path) -> None:
    root, _ = _build_bank(tmp_path, n=2)
    bank = WebDatasetBank(root)
    with pytest.raises(IndexError):
        bank[5]
