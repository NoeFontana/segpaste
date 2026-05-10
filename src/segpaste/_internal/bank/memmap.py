"""Numpy-memmap instance bank backend (ADR-0011 PR4).

On-disk layout under ``root/``:

* ``images.dat``      uint8  ``[N, 3, h, w]``
* ``alpha.dat``       bool   ``[N, 1, h, w]``
* ``classes.npy``     int64  ``[N]``
* ``embeddings.dat``  float16 ``[N, 256]`` (omitted when no embedder)
* ``meta.json``       see :mod:`segpaste._internal.bank._meta`

Random access is O(1) via mmap so the bank is callable from many DataLoader
workers without duplicating the page cache. Build cost is highest of the
three backends because crops are pre-decoded and padded to a fixed
``(h, w)``; in exchange the runtime read path has zero decode.
"""

from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import Any

import torch

from segpaste._internal.bank._meta import EMBED_DIM, base_meta, open_meta, stamp_meta
from segpaste._internal.bank.protocol import BankCrop
from segpaste._internal.imports import require_numpy

_FORMAT = "memmap"


def _images_path(root: Path) -> Path:
    return root / "images.dat"


def _alpha_path(root: Path) -> Path:
    return root / "alpha.dat"


def _classes_path(root: Path) -> Path:
    return root / "classes.npy"


def _embeddings_path(root: Path) -> Path:
    return root / "embeddings.dat"


class MemmapBank:
    """Memory-mapped :class:`InstanceBank` backend."""

    def __init__(self, root: str | Path, *, mmap_mode: str = "r") -> None:
        np = require_numpy()
        self._root = Path(root)
        meta = open_meta(self._root, expected_format=_FORMAT)
        self._n = meta.n
        self._h = meta.h
        self._w = meta.w
        self._has_embeddings = meta.has_embeddings
        self._sha256 = meta.sha256

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
                shape=(self._n, EMBED_DIM),
            )
        else:
            self._embeddings = None

        self._crop_class_ids = torch.from_numpy(self._classes.astype(np.int64).copy())
        self._class_frequencies = torch.bincount(
            self._crop_class_ids, minlength=meta.num_classes
        )

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> BankCrop:
        if idx < 0 or idx >= self._n:
            raise IndexError(idx)
        if self._images is None or self._alpha is None:
            raise RuntimeError("MemmapBank is closed")
        image = torch.from_numpy(self._images[idx].copy())
        alpha = torch.from_numpy(self._alpha[idx].copy())
        class_id = int(self._classes[idx])
        embedding = None
        if self._embeddings is not None:
            embedding = torch.from_numpy(self._embeddings[idx].copy())
        return BankCrop(
            image=image, alpha=alpha, class_id=class_id, embedding=embedding
        )

    def close(self) -> None:
        """Drop mmap references so the underlying handles can be GC'd."""
        self._images = None  # type: ignore[assignment]
        self._alpha = None  # type: ignore[assignment]
        self._embeddings = None

    def __enter__(self) -> MemmapBank:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

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
    ``embeddings: float16 [N,256]``.
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
        if emb_arr.shape != (n, EMBED_DIM):
            raise ValueError(f"embeddings shape {emb_arr.shape} != ({n}, {EMBED_DIM})")
        emb_arr.tofile(_embeddings_path(root))

    freqs = np.bincount(classes_arr, minlength=num_classes).astype(np.int64).tolist()
    meta = base_meta(
        backend_format=_FORMAT,
        n=n,
        h=h,
        w=w,
        num_classes=num_classes,
        has_embeddings=has_embeddings,
        build_seed=build_seed,
        segpaste_version=segpaste_version,
        class_frequencies=freqs,
    )
    stamp_meta(root, meta)
    return root
