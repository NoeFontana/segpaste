"""PyTorch Lightning DataModule factory (ADR-0015 §1).

Produces a :class:`lightning.pytorch.LightningDataModule` subclass bound
to a registered preset. Augmentation runs inside
``on_after_batch_transfer`` so :class:`BatchCopyPaste` executes on the
GPU side of the device transfer — matching the compile-clean contract
(ADR-0008 §D7).

``lightning`` is an optional dependency: import is deferred to factory
call time via :func:`segpaste._internal.imports.require_lightning`. The
``LightningDataModule`` subclass is constructed inside
:func:`make_segpaste_datamodule`, so the class body never runs at module
import.
"""

from __future__ import annotations

from typing import Any

from torch.utils.data import DataLoader, Dataset

from segpaste._internal.imports import require_lightning
from segpaste.augmentation.batch_copy_paste import BatchCopyPaste
from segpaste.integrations.torchvision import make_segpaste_collate_fn
from segpaste.presets import get_preset
from segpaste.types import DenseSample, PaddedBatchedDenseSample


def make_segpaste_datamodule(
    preset_name: str,
    train_dataset: Dataset[DenseSample],
    val_dataset: Dataset[DenseSample] | None = None,
    *,
    batch_size: int = 8,
    max_instances: int = 32,
    num_workers: int = 0,
) -> Any:
    """Build a :class:`LightningDataModule` bound to a registered preset.

    Train batches go through :class:`BatchCopyPaste` in
    ``on_after_batch_transfer``; validation batches are passed through
    unchanged. The augmentation kernel is constructed lazily on first
    ``setup()`` so subclassed datamodules can be moved between devices
    by Lightning's normal mechanisms.

    Raises :class:`ImportError` if ``lightning`` is not installed; the
    message includes ``pip install 'segpaste[lightning]'``.
    """
    pl = require_lightning()
    preset = get_preset(preset_name)
    collate_fn = make_segpaste_collate_fn(max_instances)

    class SegPasteDataModule(pl.LightningDataModule):  # type: ignore[misc, name-defined]
        """Preset-bound DataModule; augmentation in ``on_after_batch_transfer``."""

        def __init__(self) -> None:
            super().__init__()
            self._augment: BatchCopyPaste | None = None

        def setup(self, stage: str | None = None) -> None:
            del stage
            if self._augment is None:
                self._augment = BatchCopyPaste(preset.batch_copy_paste)

        def train_dataloader(self) -> DataLoader[DenseSample]:
            return DataLoader(
                train_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                collate_fn=collate_fn,
                shuffle=True,
            )

        def val_dataloader(self) -> DataLoader[DenseSample] | None:
            if val_dataset is None:
                return None
            return DataLoader(
                val_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                collate_fn=collate_fn,
                shuffle=False,
            )

        def on_after_batch_transfer(
            self,
            batch: PaddedBatchedDenseSample,
            dataloader_idx: int,
        ) -> PaddedBatchedDenseSample:
            del dataloader_idx
            if not self.trainer.training or self._augment is None:
                return batch
            return self._augment(batch)

    return SegPasteDataModule()
