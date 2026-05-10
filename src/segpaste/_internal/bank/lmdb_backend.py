"""LMDB instance bank backend (ADR-0011 PR5).

On-disk layout under ``root/``:

* ``data.mdb``     LMDB B-tree; keys ``b"crop:{idx:08d}"`` map to packed
                   ``(uint8 [3,h,w] image)(bool [1,h,w] alpha)(int64 class_id)
                   (optional float16 [256] embedding)`` records.
* ``meta.json``    same schema as :mod:`segpaste._internal.bank.memmap`
                   but with ``format = "lmdb"``.

Random access is true-random via ``lmdb`` (B-tree, mmap-backed); build
cost is medium (txn-write per crop). Best for single-host NVMe-scale
banks (~hundreds of GB) with many concurrent DataLoader workers.

The packed record uses fixed-stride binary so reads are constant-time
struct unpacks rather than serialized object decode. Image and alpha
strides come from ``meta.json`` (constant per bank).
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import torch

from segpaste._internal.bank.protocol import BankCrop
from segpaste._internal.imports import require_lmdb, require_numpy

_FORMAT = "lmdb"
_FORMAT_VERSION = 1
_EMBED_DIM = 256
_EMB_BYTES = _EMBED_DIM * 2  # float16


def _meta_path(root: Path) -> Path:
    return root / "meta.json"


def _hash_meta(meta: dict[str, Any]) -> str:
    payload = {k: v for k, v in meta.items() if k != "sha256"}
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def _key(idx: int) -> bytes:
    return f"crop:{idx:08d}".encode()


def _pack_record(
    image: bytes, alpha: bytes, class_id: int, embedding: bytes | None
) -> bytes:
    """Pack one crop record. Layout::

    u8[len_image]  image_bytes
    u8[len_alpha]  alpha_bytes
    i64            class_id  (little-endian)
    u8[0|emb_len]  embedding_bytes (only when bank has embeddings)
    """
    parts = [image, alpha, struct.pack("<q", int(class_id))]
    if embedding is not None:
        parts.append(embedding)
    return b"".join(parts)


def _unpack_record(
    blob: bytes, *, image_len: int, alpha_len: int, has_embeddings: bool
) -> tuple[bytes, bytes, int, bytes | None]:
    expected = image_len + alpha_len + 8 + (_EMB_BYTES if has_embeddings else 0)
    if len(blob) != expected:
        raise ValueError(
            f"corrupt LMDB record: expected {expected} bytes, got {len(blob)}"
        )
    image = blob[:image_len]
    o = image_len
    alpha = blob[o : o + alpha_len]
    o += alpha_len
    (class_id,) = struct.unpack("<q", blob[o : o + 8])
    o += 8
    embedding = blob[o : o + _EMB_BYTES] if has_embeddings else None
    return image, alpha, int(class_id), embedding


class LMDBBank:
    """LMDB-backed :class:`InstanceBank`."""

    def __init__(
        self, root: str | Path, *, max_readers: int = 256, readahead: bool = False
    ) -> None:
        np = require_numpy()
        lmdb = require_lmdb()
        self._np = np
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

        recomputed = _hash_meta(meta)
        stored = meta.get("sha256")
        if stored is not None and stored != recomputed:
            self._env.close()
            raise ValueError(
                f"{self._root}: meta.json sha256 mismatch (stored {stored!r}, "
                f"recomputed {recomputed!r})"
            )
        self._sha256 = recomputed

        self._crop_class_ids = torch.tensor(
            meta["per_crop_class_ids"], dtype=torch.int64
        )
        self._class_frequencies = torch.bincount(
            self._crop_class_ids, minlength=self._num_classes
        )

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> BankCrop:
        if idx < 0 or idx >= self._n:
            raise IndexError(idx)
        with self._env.begin(buffers=False) as txn:
            blob = txn.get(_key(idx))
        if blob is None:
            raise KeyError(f"missing crop key for idx={idx}")
        image_b, alpha_b, class_id, emb_b = _unpack_record(
            bytes(blob),
            image_len=self._image_len,
            alpha_len=self._alpha_len,
            has_embeddings=self._has_embeddings,
        )
        np = self._np
        image = torch.from_numpy(
            np.frombuffer(image_b, dtype=np.uint8).reshape(3, self._h, self._w).copy()
        )
        alpha = torch.from_numpy(
            np.frombuffer(alpha_b, dtype=np.bool_).reshape(1, self._h, self._w).copy()
        )
        embedding = None
        if emb_b is not None:
            embedding = torch.from_numpy(np.frombuffer(emb_b, dtype=np.float16).copy())
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

    ``images: uint8 [N,3,h,w]``, ``alpha: bool [N,1,h,w]``, ``classes: int64 [N]``,
    optional ``embeddings: float16 [N,256]``. Per-crop layout follows
    :func:`_pack_record`. ``map_size_bytes`` is the LMDB map size — the
    default is sparse and won't physically allocate.
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
    if has_embeddings:
        emb_arr = np.ascontiguousarray(embeddings, dtype=np.float16)
    else:
        emb_arr = None

    env = lmdb.open(
        str(root), map_size=map_size_bytes, subdir=True, max_dbs=1, lock=False
    )
    try:
        with env.begin(write=True) as txn:
            for i in range(n):
                emb_bytes = emb_arr[i].tobytes() if emb_arr is not None else None
                blob = _pack_record(
                    images_arr[i].tobytes(),
                    alpha_arr[i].tobytes(),
                    int(classes_arr[i]),
                    emb_bytes,
                )
                txn.put(_key(i), blob)
    finally:
        env.close()

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
        # Carrying per-crop class ids in meta avoids an O(N) txn at open time.
        "per_crop_class_ids": [int(x) for x in classes_arr.tolist()],
    }
    meta["sha256"] = _hash_meta(meta)
    with _meta_path(root).open("w") as fh:
        json.dump(meta, fh, sort_keys=True, indent=2)
    return root
