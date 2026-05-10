"""Bank-of-instance-crops protocol + crop record (ADR-0011)."""

from __future__ import annotations

from typing import NamedTuple, Protocol, runtime_checkable

import torch


class BankCrop(NamedTuple):
    """A single instance crop from an :class:`InstanceBank`.

    ``image`` and ``alpha`` are the decoded crop and its instance mask at
    the bank's fixed ``(h, w)``. ``class_id`` is a Python int so default
    collate stacks crops without per-element type promotion. ``embedding``
    is a 256-d ``float16`` vector (e.g. CLIP) when the bank was built with
    embeddings, otherwise ``None``; banks expose ``has_embeddings`` as a
    once-per-bank flag, never per-crop.
    """

    image: torch.Tensor  # uint8  [3, h, w]
    alpha: torch.Tensor  # bool   [1, h, w]
    class_id: int  # int64 scalar
    embedding: torch.Tensor | None  # float16 [256] or None


@runtime_checkable
class InstanceBank(Protocol):
    """Read-only sequence of class-labeled instance crops.

    Concrete backends (``MemmapBank``, ``LMDBBank``, ``WebDatasetBank``)
    live under :mod:`segpaste._internal.bank` until promotion. The
    Protocol is the only public name; users construct backends via the
    ``scripts/build_instance_bank.py`` CLI (PR5) or import the backend
    class directly from ``segpaste._internal.bank``.

    Implementations must be safe to call from DataLoader workers — i.e.
    re-entrant after ``__init__`` and free of un-pickle-able state — so
    ``num_workers > 0`` is supported.
    """

    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> BankCrop: ...

    @property
    def class_frequencies(self) -> torch.Tensor:
        """``int64 [num_classes]`` count of crops per class (zero-indexed)."""
        ...

    @property
    def crop_class_ids(self) -> torch.Tensor:
        """``int64 [N]`` class id per crop. Zero-copy on memmap backends;
        loaded once at open. Lets :class:`BankSampler` build per-crop
        weights without an O(N) pass through ``__getitem__``."""
        ...

    @property
    def crop_size(self) -> tuple[int, int]:
        """``(h, w)`` after preprocessing — the same for every crop."""
        ...

    @property
    def has_embeddings(self) -> bool:
        """Whether ``BankCrop.embedding`` is populated for every crop."""
        ...

    @property
    def version(self) -> str:
        """Stable ``{format}@{sha256[:12]}`` identifier for cache keys."""
        ...
