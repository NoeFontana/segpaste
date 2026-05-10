"""WebDataset (sharded tar) instance bank backend (ADR-0011 PR6).

On-disk layout under ``root/``:

* ``shard-{NNNNNN}.tar``  shard files, each holding contiguous crop
                          records ``crop_{idx:08d}.bin`` packed in the
                          same binary layout used by ``LMDBBank``.
* ``index.parquet``       columns ``(idx, shard_id, member_name,
                          class_id)`` so ``BankSampler`` can open in
                          O(1) and so downstream filters (eg by class)
                          can run as a parquet scan.
* ``meta.json``           ``format = "webdataset"`` + the same metadata
                          fields as the other backends.

Random access is via ``tarfile.TarFile.extractfile`` against an index
built once per shard. Each shard is opened lazily and cached on the
``WebDatasetBank`` instance so subsequent reads from the same shard
hit the OS page cache.
"""

from __future__ import annotations

import hashlib
import json
import struct
import tarfile
from pathlib import Path
from typing import Any

import torch

from segpaste._internal.bank.protocol import BankCrop
from segpaste._internal.imports import require_numpy, require_pyarrow

_FORMAT = "webdataset"
_FORMAT_VERSION = 1
_EMBED_DIM = 256
_EMB_BYTES = _EMBED_DIM * 2  # float16


def _meta_path(root: Path) -> Path:
    return root / "meta.json"


def _index_path(root: Path) -> Path:
    return root / "index.parquet"


def _shard_path(root: Path, shard_id: int) -> Path:
    return root / f"shard-{shard_id:06d}.tar"


def _hash_meta(meta: dict[str, Any]) -> str:
    payload = {k: v for k, v in meta.items() if k != "sha256"}
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def _pack_record(
    image: bytes, alpha: bytes, class_id: int, embedding: bytes | None
) -> bytes:
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
            f"corrupt webdataset record: expected {expected} bytes, got {len(blob)}"
        )
    image = blob[:image_len]
    o = image_len
    alpha = blob[o : o + alpha_len]
    o += alpha_len
    (class_id,) = struct.unpack("<q", blob[o : o + 8])
    o += 8
    embedding = blob[o : o + _EMB_BYTES] if has_embeddings else None
    return image, alpha, int(class_id), embedding


def _member_name(idx: int) -> str:
    return f"crop_{idx:08d}.bin"


class WebDatasetBank:
    """Sharded-tar :class:`InstanceBank`. Optimized for streaming + filter."""

    def __init__(self, root: str | Path) -> None:
        np = require_numpy()
        pa = require_pyarrow()
        self._np = np
        self._pa = pa
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

        # Lazy shard cache — opened on first read, kept until close.
        self._shards: dict[int, tarfile.TarFile] = {}

        # Eagerly load index.parquet — small and avoids worker contention later.
        from pyarrow import parquet  # type: ignore[import-untyped]

        table = parquet.read_table(_index_path(self._root))
        self._idx = table.column("idx").to_numpy(zero_copy_only=False)
        self._shard_id = table.column("shard_id").to_numpy(zero_copy_only=False)
        self._member_names: list[str] = table.column("member_name").to_pylist()
        self._index_class_ids = table.column("class_id").to_numpy(zero_copy_only=False)
        # The parquet table is the source of truth; assert consistency with meta.
        if int(self._idx.shape[0]) != self._n:
            raise ValueError(
                f"index.parquet size {self._idx.shape[0]} != meta.num_crops {self._n}"
            )

        self._crop_class_ids = torch.from_numpy(
            self._index_class_ids.astype(np.int64).copy()
        )
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

    def _shard(self, shard_id: int) -> tarfile.TarFile:
        cached = self._shards.get(shard_id)
        if cached is None:
            cached = tarfile.open(  # noqa: SIM115 — held in self._shards
                _shard_path(self._root, shard_id), mode="r"
            )
            self._shards[shard_id] = cached
        return cached

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> BankCrop:
        if idx < 0 or idx >= self._n:
            raise IndexError(idx)
        shard_id = int(self._shard_id[idx])
        member = self._member_names[idx]
        tar = self._shard(shard_id)
        fh = tar.extractfile(member)
        if fh is None:
            raise KeyError(f"missing tar member for idx={idx}: {member!r}")
        try:
            blob = fh.read()
        finally:
            fh.close()
        image_b, alpha_b, class_id, emb_b = _unpack_record(
            blob,
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


def write_webdataset_bank(
    root: str | Path,
    *,
    images: Any,
    alpha: Any,
    classes: Any,
    num_classes: int,
    embeddings: Any | None = None,
    build_seed: int = 0,
    segpaste_version: str = "0",
    crops_per_shard: int = 4096,
) -> Path:
    """Write a sharded webdataset bank under ``root``.

    Splits the input across multiple ``shard-{NNNNNN}.tar`` files of at
    most ``crops_per_shard`` crops each. Index parquet records the
    ``(idx, shard_id, member_name, class_id)`` mapping so downstream
    workers can mass-filter by class without opening shards.
    """
    np = require_numpy()
    pa = require_pyarrow()
    from pyarrow import parquet  # type: ignore[import-untyped]

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

    idx_col: list[int] = []
    shard_col: list[int] = []
    member_col: list[str] = []
    class_col: list[int] = []
    num_shards = (n + crops_per_shard - 1) // crops_per_shard
    for shard_id in range(num_shards):
        start = shard_id * crops_per_shard
        end = min(start + crops_per_shard, n)
        with tarfile.open(_shard_path(root, shard_id), mode="w") as tar:
            for i in range(start, end):
                emb_bytes = emb_arr[i].tobytes() if emb_arr is not None else None
                blob = _pack_record(
                    images_arr[i].tobytes(),
                    alpha_arr[i].tobytes(),
                    int(classes_arr[i]),
                    emb_bytes,
                )
                name = _member_name(i)
                info = tarfile.TarInfo(name=name)
                info.size = len(blob)
                # Stable mtime so re-builds are byte-identical.
                info.mtime = 0
                from io import BytesIO

                tar.addfile(info, BytesIO(blob))
                idx_col.append(i)
                shard_col.append(shard_id)
                member_col.append(name)
                class_col.append(int(classes_arr[i]))

    table = pa.table(
        {
            "idx": pa.array(idx_col, type=pa.int64()),
            "shard_id": pa.array(shard_col, type=pa.int32()),
            "member_name": pa.array(member_col, type=pa.string()),
            "class_id": pa.array(class_col, type=pa.int64()),
        }
    )
    parquet.write_table(table, _index_path(root))

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
        "num_shards": int(num_shards),
        "crops_per_shard": int(crops_per_shard),
    }
    meta["sha256"] = _hash_meta(meta)
    with _meta_path(root).open("w") as fh:
        json.dump(meta, fh, sort_keys=True, indent=2)
    return root
