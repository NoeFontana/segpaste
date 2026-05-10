"""Smoke + integration tests for :class:`BankSource` (ADR-0011 PR7)."""

from __future__ import annotations

import pytest
import torch
from torchvision import tv_tensors

from segpaste import BankSource, BatchCopyPaste, IntraBatchSource
from segpaste._internal.gpu.batched_placement import BatchedPlacementConfig
from segpaste.augmentation.batch_copy_paste import BatchCopyPasteConfig
from segpaste.augmentation.source_config import (
    BankSourceConfig,
    IntraBatchSourceConfig,
    build_source_strategy,
)
from segpaste.types import BatchedDenseSample, DenseSample, InstanceMask

H = W = 32
B = 2
K = 4


def _padded():
    samples: list[DenseSample] = []
    for i in range(B):
        gen = torch.Generator().manual_seed(i)
        n = 2
        masks = torch.zeros(n, H, W, dtype=torch.bool)
        boxes_list: list[list[int]] = []
        for j in range(n):
            x1, y1 = 4 + j * 6, 4 + j * 6
            x2, y2 = x1 + 6, y1 + 6
            masks[j, y1:y2, x1:x2] = True
            boxes_list.append([x1, y1, x2, y2])
        samples.append(
            DenseSample(
                image=tv_tensors.Image(torch.rand(3, H, W, generator=gen)),
                boxes=tv_tensors.BoundingBoxes(  # pyright: ignore[reportCallIssue]
                    torch.tensor(boxes_list, dtype=torch.float32),
                    format=tv_tensors.BoundingBoxFormat.XYXY,
                    canvas_size=(H, W),
                ),
                labels=torch.arange(1, n + 1, dtype=torch.int64),
                instance_ids=torch.arange(n, dtype=torch.int32),
                instance_masks=InstanceMask(masks),
            )
        )
    return BatchedDenseSample.from_samples(samples).to_padded(max_instances=K)


def _bank_batch(b: int = B, k_bank: int = 3, h: int = 8, w: int = 8) -> torch.Tensor:
    """Synthetic bank batch ``[B, K_bank, 5, h, w]`` with class-id channel set."""
    gen = torch.Generator().manual_seed(123)
    rgb = torch.rand((b, k_bank, 3, h, w), generator=gen)
    alpha = torch.zeros((b, k_bank, 1, h, w))
    alpha[..., 1:7, 1:7] = 1.0
    classes = torch.full(
        (b, k_bank, 1, h, w), float(7), dtype=torch.float32
    )  # class id = 7
    return torch.cat([rgb, alpha, classes], dim=2)


def test_build_source_strategy_dispatches_to_bank_source() -> None:
    strategy = build_source_strategy(
        BankSourceConfig(),
        BatchedPlacementConfig(),  # pyright: ignore[reportCallIssue]
    )
    assert isinstance(strategy, BankSource)


def test_build_source_strategy_dispatches_to_intra_batch() -> None:
    strategy = build_source_strategy(IntraBatchSourceConfig(), BatchedPlacementConfig())
    assert isinstance(strategy, IntraBatchSource)


def test_bank_source_requires_bank_batch_before_forward() -> None:
    strategy = BankSource()
    target = _padded()
    with pytest.raises(RuntimeError, match="set_bank_batch"):
        strategy.sample(target, valid_extent=None, source_eligible=None, generator=None)


def test_bank_source_rejects_wrong_shape() -> None:
    strategy = BankSource()
    bad = torch.zeros((B, 3, 5, 8))  # 4D
    with pytest.raises(ValueError, match=r"\[B, K_bank"):
        strategy.set_bank_batch(bad)


def test_bank_source_sample_returns_canvas_view_and_placement() -> None:
    strategy = BankSource()
    strategy.set_bank_batch(_bank_batch())
    target = _padded()
    source_view, placement = strategy.sample(
        target,
        valid_extent=None,
        source_eligible=None,
        generator=torch.Generator().manual_seed(0),
    )
    # Source view is row-aligned with target and has K_source = 1.
    assert source_view.batch_size == target.batch_size
    assert source_view.images.shape[-2:] == target.images.shape[-2:]
    assert source_view.max_instances == 1
    assert source_view.labels.shape == (B, 1)
    # All source labels == 7 (class-id channel value).
    assert torch.all(source_view.labels == 7)
    # Placement shapes.
    assert placement.source_idx.shape == (B,)
    assert torch.equal(placement.source_idx, torch.arange(B, dtype=torch.int64))
    assert placement.paste_valid.shape == (B, 1)
    assert placement.translate.shape == (B, 2)
    assert placement.scale.shape == (B,)
    assert placement.hflip.shape == (B,)


def test_batch_copy_paste_with_bank_source_produces_padded_output() -> None:
    bank_strategy = BankSource()
    bank_strategy.set_bank_batch(_bank_batch())
    module = BatchCopyPaste(
        BatchCopyPasteConfig(source=BankSourceConfig()),  # pyright: ignore[reportCallIssue]
        source_strategy=bank_strategy,
    )
    target = _padded()
    out = module(target, generator=torch.Generator().manual_seed(0))
    # Same canvas, same K, instance_valid is bool.
    assert out.images.shape == target.images.shape
    assert out.max_instances == K
    assert out.instance_valid.dtype == torch.bool


def test_intra_batch_default_config_path_unaffected() -> None:
    """The default ``BatchCopyPasteConfig()`` still resolves to IntraBatchSource."""
    module = BatchCopyPaste()
    assert isinstance(module.source_strategy, IntraBatchSource)


def test_default_dispatch_does_not_fail_without_bank_batch() -> None:
    """Default config doesn't need a bank_batch — IntraBatchSource path."""
    module = BatchCopyPaste()
    target = _padded()
    out = module(target, generator=torch.Generator().manual_seed(0))
    assert out.batch_size == B
