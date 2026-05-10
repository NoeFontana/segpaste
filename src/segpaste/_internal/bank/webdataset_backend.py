"""WebDataset (sharded tar) instance bank backend (ADR-0011 PR6).

On-disk layout under ``root/``:

* ``shard-{NNNNNN}.tar``  shard files, each holding contiguous crop records
                          ``crop_{idx:08d}.bin`` packed as in
                          :mod:`segpaste._internal.bank._record`.
* ``index.parquet``       columns ``(idx, shard_id, member_name, class_id)``
                          — opens the bank in O(1) and supports class
                          filtering as a parquet scan.
* ``meta.json``           ``format = "webdataset"`` plus the same metadata
                          fields as the other backends.

Random access is via :func:`tarfile.TarFile.extractfile` against an index
built once per shard. Each shard is opened lazily and cached on the
``WebDatasetBank`` instance so subsequent reads from the same shard
hit the OS page cache.
"""

from __future__ import annotations

import contextlib
import tarfile
from io import BytesIO
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
from segpaste._internal.imports import require_numpy, require_pyarrow

_FORMAT = "webdataset"


def _index_path(root: Path) -> Path:
    return root / "index.parquet"


def _shard_path(root: Path, shard_id: int) -> Path:
    return root / f"shard-{shard_id:06d}.tar"


def _member_name(idx: int) -> str:
    return f"crop_{idx:08d}.bin"


class WebDatasetBank:
    """Sharded-tar :class:`InstanceBank`. Optimized for streaming + filter."""

    def __init__(self, root: str | Path) -> None:
        np = require_numpy()
        require_pyarrow()
        self._root = Path(root)
        meta = open_meta(self._root, expected_format=_FORMAT)
        self._n = meta.n
        self._h = meta.h
        self._w = meta.w
        self._has_embeddings = meta.has_embeddings
        self._sha256 = meta.sha256
        self._image_len = 3 * self._h * self._w
        self._alpha_len = self._h * self._w

        self._shards: dict[int, tarfile.TarFile] = {}

        from pyarrow import parquet  # type: ignore[import-untyped]

        table = parquet.read_table(_index_path(self._root))
        if table.num_rows != self._n:
            raise ValueError(
                f"index.parquet size {table.num_rows} != meta.num_crops {self._n}"
            )
        self._shard_id = table.column("shard_id").to_numpy(zero_copy_only=False)
        self._member_names: list[str] = table.column("member_name").to_pylist()
        index_class_ids = table.column("class_id").to_numpy(zero_copy_only=False)

        self._crop_class_ids = torch.from_numpy(index_class_ids.astype(np.int64).copy())
        self._class_frequencies = torch.bincount(
            self._crop_class_ids, minlength=meta.num_classes
        )

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
        image_b, alpha_b, class_id, emb_b = unpack_record(
            blob,
            image_len=self._image_len,
            alpha_len=self._alpha_len,
            has_embeddings=self._has_embeddings,
            backend_format=_FORMAT,
        )
        return decode_crop(image_b, alpha_b, class_id, emb_b, h=self._h, w=self._w)

    def close(self) -> None:
        """Close every cached shard. Safe to call multiple times."""
        for tar in self._shards.values():
            tar.close()
        self._shards.clear()

    def __enter__(self) -> WebDatasetBank:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def __del__(self) -> None:
        # Suppress all exceptions — destructors must not raise.
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

    Splits the input across ``shard-{NNNNNN}.tar`` files of at most
    ``crops_per_shard`` crops each. Index parquet records the
    ``(idx, shard_id, member_name, class_id)`` mapping so workers can
    mass-filter by class without opening shards.
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
    emb_arr = (
        np.ascontiguousarray(embeddings, dtype=np.float16) if has_embeddings else None
    )

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
                blob = pack_record(
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
    meta["num_shards"] = int(num_shards)
    meta["crops_per_shard"] = int(crops_per_shard)
    stamp_meta(root, meta)
    return root
