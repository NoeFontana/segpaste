"""External instance-crop bank machinery (ADR-0011).

Public protocol :class:`InstanceBank` + concrete backends (memmap, lmdb,
webdataset) and the :class:`BankSampler` that drives a class-balanced
DataLoader. Worker-side decode + sampling, so the GPU forward sees only
pre-staged tensors and the empty compile allow-list stays empty.
"""

from __future__ import annotations

from segpaste._internal.bank.protocol import BankCrop, InstanceBank

__all__ = ["BankCrop", "InstanceBank"]
