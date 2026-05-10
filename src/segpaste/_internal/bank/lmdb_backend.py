"""LMDB instance bank backend (ADR-0011 PR5).

On-disk layout under ``root/``:

* ``data.mdb``     LMDB B-tree; keys ``b"crop:{idx:08d}"`` map to packed
                   crop records (see :mod:`segpaste._internal.bank._record`).
* ``meta.json``    same schema as :mod:`segpaste._internal.bank.memmap`
                   but with ``format = "lmdb"`` and a ``per_crop_class_ids``
                   list so :class:`BankSampler` opens in O(1).

Random access is true-random via ``lmdb`` (B-tree, mmap-backed); build
cost is medium (txn-write per crop). Best for single-host NVMe-scale
banks with many concurrent DataLoader workers.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from types import TracebackType
from typing import Any

import torch

from segpaste._internal.bank._meta import (
    base_meta,
    open_meta,
    stamp_meta,
)
from segpaste._internal.bank._record import (
    decode_crop,
    pack_record,
    unpack_record,
)
from segpaste._internal.bank.protocol import BankCrop
from segpaste._internal.imports import require_lmdb, require_numpy

_FORMAT = "lmdb"


def _key(idx: int) -> bytes:
    return f"crop:{idx:08d}".encode()


class LMDBBank:
    """LMDB-backed :class:`InstanceBank`."""

    def __init__(
        self, root: str | Path, *, max_readers: int = 256, readahead: bool = False
    ) -> None:
        require_numpy()
        lmdb = require_lmdb()
        self._root = Path(root)
        meta = open_meta(self._root, expected_format=_FORMAT)
        self._n = meta.n
        self._h = meta.h
        self._w = meta.w
        self._has_embeddings = meta.has_embeddings
        self._sha256 = meta.sha256
        self._image_len = 3 * self._h * self._w
        self._alpha_len = self._h * self._w

        self._env = lmdb.open(
            str(self._root),
            readonly=True,
            lock=False,
            max_readers=max_readers,
            readahead=readahead,
            subdir=True,
        )

        self._crop_class_ids = torch.tensor(
            meta.raw["per_crop_class_ids"], dtype=torch.int64
        )
        self._class_frequencies = torch.bincount(
            self._crop_class_ids, minlength=meta.num_classes
        )

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> BankCrop:
        if idx < 0 or idx >= self._n:
            raise IndexError(idx)
        env = self._env
        if env is None:
            raise RuntimeError("LMDBBank is closed")
        # ``buffers=True`` returns ``memoryview``s into LMDB-owned pages —
        # ``decode_crop`` copies into Python-owned tensors before this scope
        # exits, so releasing the txn after the with-block is safe.
        with env.begin(buffers=True) as txn:
            blob = txn.get(_key(idx))
            if blob is None:
                raise KeyError(f"missing crop key for idx={idx}")
            image_b, alpha_b, class_id, emb_b = unpack_record(
                bytes(blob),
                image_len=self._image_len,
                alpha_len=self._alpha_len,
                has_embeddings=self._has_embeddings,
                backend_format=_FORMAT,
            )
        return decode_crop(image_b, alpha_b, class_id, emb_b, h=self._h, w=self._w)

    def close(self) -> None:
        """Close the LMDB environment. Safe to call multiple times."""
        env = getattr(self, "_env", None)
        if env is not None:
            env.close()
            self._env = None  # type: ignore[assignment]

    def __enter__(self) -> LMDBBank:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def __del__(self) -> None:
        # LMDB env holds an mmap that needs explicit close; suppress all
        # exceptions because destructors must not raise.
        with contextlib.suppress(Exception):
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


def write_lmdb_bank(
    root: str | Path,
    *,
    images: Any,
    alpha: Any,
    classes: Any,
    num_classes: int,
    embeddings: Any | None = None,
    build_seed: int = 0,
    segpaste_version: str = "0",
    map_size_bytes: int = 1 << 36,  # 64 GiB sparse default — actual usage is tiny
) -> Path:
    """Write an LMDB bank under ``root``.

    See :func:`segpaste._internal.bank.memmap.write_memmap_bank` for input
    shape conventions.
    """
    np = require_numpy()
    lmdb = require_lmdb()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    images_arr = np.ascontiguousarray(images, dtype=np.uint8)
    alpha_arr = np.ascontiguousarray(alpha, dtype=np.bool_)
    classes_arr = np.ascontiguousarray(classes, dtype=np.int64)
    n, _, h, w = images_arr.shape

    has_embeddings = embeddings is not None
    emb_arr = (
        np.ascontiguousarray(embeddings, dtype=np.float16) if has_embeddings else None
    )

    env = lmdb.open(
        str(root), map_size=map_size_bytes, subdir=True, max_dbs=1, lock=False
    )
    try:
        with env.begin(write=True) as txn:
            for i in range(n):
                emb_bytes = emb_arr[i].tobytes() if emb_arr is not None else None
                blob = pack_record(
                    images_arr[i].tobytes(),
                    alpha_arr[i].tobytes(),
                    int(classes_arr[i]),
                    emb_bytes,
                )
                txn.put(_key(i), blob)
    finally:
        env.close()

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
    # Carrying per-crop class ids in meta avoids an O(N) txn at open time.
    meta["per_crop_class_ids"] = [int(x) for x in classes_arr.tolist()]
    stamp_meta(root, meta)
    return root
