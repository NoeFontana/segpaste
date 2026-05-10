"""Shared packed-binary crop record format (ADR-0011).

Both :class:`LMDBBank` and :class:`WebDatasetBank` store crops as a
fixed-stride packed binary record::

    u8[3*h*w]   image_bytes      (uint8 [3, h, w], C-contiguous)
    u8[h*w]     alpha_bytes      (bool  [1, h, w], C-contiguous)
    i8[8]       class_id         (int64, little-endian)
    u8[512]     embedding_bytes  (float16 [256], present only if bank has embeddings)

The format is shared so wire-compatibility between the two backends is a
single source of truth. ``MemmapBank`` doesn't use packed records — its
arrays are mmap-direct.
"""

from __future__ import annotations

import struct
from typing import Any

import torch

from segpaste._internal.bank._meta import EMB_BYTES
from segpaste._internal.bank.protocol import BankCrop
from segpaste._internal.imports import require_numpy


def pack_record(
    image: bytes, alpha: bytes, class_id: int, embedding: bytes | None
) -> bytes:
    parts = [image, alpha, struct.pack("<q", int(class_id))]
    if embedding is not None:
        parts.append(embedding)
    return b"".join(parts)


def record_size(*, image_len: int, alpha_len: int, has_embeddings: bool) -> int:
    return image_len + alpha_len + 8 + (EMB_BYTES if has_embeddings else 0)


def unpack_record(
    blob: bytes,
    *,
    image_len: int,
    alpha_len: int,
    has_embeddings: bool,
    backend_format: str,
) -> tuple[bytes, bytes, int, bytes | None]:
    expected = record_size(
        image_len=image_len, alpha_len=alpha_len, has_embeddings=has_embeddings
    )
    if len(blob) != expected:
        raise ValueError(
            f"corrupt {backend_format} record: expected {expected} bytes, "
            f"got {len(blob)}"
        )
    image = blob[:image_len]
    o = image_len
    alpha = blob[o : o + alpha_len]
    o += alpha_len
    (class_id,) = struct.unpack("<q", blob[o : o + 8])
    o += 8
    embedding = blob[o : o + EMB_BYTES] if has_embeddings else None
    return image, alpha, int(class_id), embedding


def decode_crop(
    image_bytes: bytes,
    alpha_bytes: bytes,
    class_id: int,
    embedding_bytes: bytes | None,
    *,
    h: int,
    w: int,
) -> BankCrop:
    """Decode raw packed bytes into a :class:`BankCrop`.

    Each tensor is materialized via ``np.frombuffer(...).copy()`` so the
    underlying storage is owned by Python — safe to release the source
    transaction or mmap page after this returns.
    """
    np: Any = require_numpy()
    image = torch.from_numpy(
        np.frombuffer(image_bytes, dtype=np.uint8).reshape(3, h, w).copy()
    )
    alpha = torch.from_numpy(
        np.frombuffer(alpha_bytes, dtype=np.bool_).reshape(1, h, w).copy()
    )
    embedding = None
    if embedding_bytes is not None:
        embedding = torch.from_numpy(
            np.frombuffer(embedding_bytes, dtype=np.float16).copy()
        )
    return BankCrop(image=image, alpha=alpha, class_id=class_id, embedding=embedding)


__all__ = ["decode_crop", "pack_record", "record_size", "unpack_record"]
