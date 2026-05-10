"""Numpy-memmap instance bank backend (ADR-0011 PR4).

On-disk layout under ``root/``:

* ``images.dat``      uint8  ``[N, 3, h, w]``
* ``alpha.dat``       bool   ``[N, 1, h, w]``
* ``classes.npy``     int64  ``[N]``
* ``embeddings.dat``  float16 ``[N, 256]`` (omitted when no embedder)
* ``meta.json``       ``{format, num_crops, crop_h, crop_w, num_classes,
                         has_embeddings, segpaste_version, build_seed,
                         class_frequencies, sha256}``

Random access is O(1) via mmap so the bank is callable from many DataLoader
workers without duplicating the page cache. Build cost is highest of the
three backends because crops are pre-decoded and padded to a fixed
``(h, w)``; in exchange the runtime read path has zero decode.

The ``sha256`` field hashes ``meta.json`` minus itself; ``version`` returns
``"memmap@{sha256[:12]}"`` for cache keys.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from segpaste._internal.bank.protocol import BankCrop
from segpaste._internal.imports import require_numpy

_FORMAT = "memmap"
_FORMAT_VERSION = 1
_EMBED_DIM = 256


def _meta_path(root: Path) -> Path:
    return root / "meta.json"


def _images_path(root: Path) -> Path:
    return root / "images.dat"


def _alpha_path(root: Path) -> Path:
    return root / "alpha.dat"


def _classes_path(root: Path) -> Path:
    return root / "classes.npy"


def _embeddings_path(root: Path) -> Path:
    return root / "embeddings.dat"


def _hash_meta(meta: dict[str, Any]) -> str:
    payload = {k: v for k, v in meta.items() if k != "sha256"}
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


class MemmapBank:
    """Memory-mapped :class:`InstanceBank` backend.

    Open with :meth:`open` (the canonical entry point) or directly via
    ``MemmapBank(root)``. The ``mmap_mode`` defaults to read-only ``"r"``
    so multiple workers share the page cache.
    """

    def __init__(self, root: str | Path, *, mmap_mode: str = "r") -> None:
        np = require_numpy()
        self._root = Path(root)
        with _meta_path(self._root).open("r") as fh:
            meta = json.load(fh)
        if meta.get("format") != _FORMAT:
            raise ValueError(
                f"{self._root}: meta format {meta.get('format')!r} != {_FORMAT!r}"
            )
        if meta.get("format_version") != _FORMAT_VERSION:
            raise ValueError(
                f"{self._root}: format_version {meta.get('format_version')} "
                f"!= {_FORMAT_VERSION}"
            )
        self._meta = meta
        self._n: int = int(meta["num_crops"])
        self._h: int = int(meta["crop_h"])
        self._w: int = int(meta["crop_w"])
        self._num_classes: int = int(meta["num_classes"])
        self._has_embeddings: bool = bool(meta["has_embeddings"])

        self._images = np.memmap(
            _images_path(self._root),
            dtype=np.uint8,
            mode=mmap_mode,
            shape=(self._n, 3, self._h, self._w),
        )
        self._alpha = np.memmap(
            _alpha_path(self._root),
            dtype=np.bool_,
            mode=mmap_mode,
            shape=(self._n, 1, self._h, self._w),
        )
        self._classes = np.load(_classes_path(self._root), mmap_mode=mmap_mode)
        if self._has_embeddings:
            self._embeddings = np.memmap(
                _embeddings_path(self._root),
                dtype=np.float16,
                mode=mmap_mode,
                shape=(self._n, _EMBED_DIM),
            )
        else:
            self._embeddings = None

        self._crop_class_ids = torch.from_numpy(self._classes.astype(np.int64).copy())
        self._class_frequencies = torch.bincount(
            self._crop_class_ids, minlength=self._num_classes
        )

        recomputed = _hash_meta(meta)
        stored = meta.get("sha256")
        if stored is not None and stored != recomputed:
            raise ValueError(
                f"{self._root}: meta.json sha256 mismatch (stored {stored!r}, "
                f"recomputed {recomputed!r})"
            )
        self._sha256 = recomputed

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> BankCrop:
        if idx < 0 or idx >= self._n:
            raise IndexError(idx)
        image = torch.from_numpy(self._images[idx].copy())
        alpha = torch.from_numpy(self._alpha[idx].copy())
        class_id = int(self._classes[idx])
        if self._embeddings is not None:
            embedding = torch.from_numpy(self._embeddings[idx].copy())
        else:
            embedding = None
        return BankCrop(
            image=image, alpha=alpha, class_id=class_id, embedding=embedding
        )

    @property
    def class_frequencies(self) -> torch.Tensor:
        return self._class_frequencies

    @property
    def crop_class_ids(self) -> torch.Tensor:
        return self._crop_class_ids

    @property
    def crop_size(self) -> tuple[int, int]:
        return (self._h, self._w)

    @property
    def has_embeddings(self) -> bool:
        return self._has_embeddings

    @property
    def version(self) -> str:
        return f"{_FORMAT}@{self._sha256[:12]}"


def write_memmap_bank(
    root: str | Path,
    *,
    images: Any,
    alpha: Any,
    classes: Any,
    num_classes: int,
    embeddings: Any | None = None,
    build_seed: int = 0,
    segpaste_version: str = "0",
) -> Path:
    """Write a memmap bank under ``root``.

    Parameters are pre-shaped numpy arrays — ``images: uint8 [N,3,h,w]``,
    ``alpha: bool [N,1,h,w]``, ``classes: int64 [N]``, optional
    ``embeddings: float16 [N,256]``. Used by the build script (PR5) and
    the test suite. Returns ``root``.
    """
    np = require_numpy()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    images_arr = np.ascontiguousarray(images, dtype=np.uint8)
    alpha_arr = np.ascontiguousarray(alpha, dtype=np.bool_)
    classes_arr = np.ascontiguousarray(classes, dtype=np.int64)
    if images_arr.ndim != 4 or images_arr.shape[1] != 3:
        raise ValueError(f"images must be [N, 3, h, w], got {images_arr.shape}")
    if alpha_arr.shape != (
        images_arr.shape[0],
        1,
        images_arr.shape[2],
        images_arr.shape[3],
    ):
        raise ValueError(
            f"alpha shape {alpha_arr.shape} mismatches images {images_arr.shape}"
        )
    if classes_arr.shape != (images_arr.shape[0],):
        raise ValueError(
            f"classes shape {classes_arr.shape} mismatches N={images_arr.shape[0]}"
        )

    n, _, h, w = images_arr.shape

    images_arr.tofile(_images_path(root))
    alpha_arr.tofile(_alpha_path(root))
    np.save(_classes_path(root), classes_arr)

    has_embeddings = embeddings is not None
    if has_embeddings:
        emb_arr = np.ascontiguousarray(embeddings, dtype=np.float16)
        if emb_arr.shape != (n, _EMBED_DIM):
            raise ValueError(f"embeddings shape {emb_arr.shape} != ({n}, {_EMBED_DIM})")
        emb_arr.tofile(_embeddings_path(root))

    freqs = np.bincount(classes_arr, minlength=num_classes).astype(np.int64)
    meta: dict[str, Any] = {
        "format": _FORMAT,
        "format_version": _FORMAT_VERSION,
        "num_crops": int(n),
        "crop_h": int(h),
        "crop_w": int(w),
        "num_classes": int(num_classes),
        "has_embeddings": bool(has_embeddings),
        "segpaste_version": str(segpaste_version),
        "build_seed": int(build_seed),
        "class_frequencies": [int(x) for x in freqs.tolist()],
    }
    meta["sha256"] = _hash_meta(meta)
    with _meta_path(root).open("w") as fh:
        json.dump(meta, fh, sort_keys=True, indent=2)
    return root
