"""Compile-clean gate for the :class:`BankSource` forward path (ADR-0011 PR7).

Mirrors ``tests/test_compile_clean.py`` but parametrizes over a
:class:`BankSource`-driven ``BatchCopyPaste``. Same empty allow-list —
the bank batch is pre-staged so disk I/O and worker-side sampling stay
out of the compiled graph.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.compile_explain import (
    build_fixture,
    explain_breaks,
    load_allowlist,
    partition,
)
from segpaste import BankSource, BatchCopyPaste
from segpaste.augmentation.batch_copy_paste import BatchCopyPasteConfig
from segpaste.augmentation.source_config import BankSourceConfig

pytest.importorskip("torch._dynamo")

ALLOWLIST_PATH = Path(__file__).parents[1] / "scripts" / "compile_allowlist.txt"


def _bank_batch(b: int, k_bank: int = 3, h: int = 8, w: int = 8) -> torch.Tensor:
    rgb = torch.zeros((b, k_bank, 3, h, w), dtype=torch.float32)
    alpha = torch.zeros((b, k_bank, 1, h, w), dtype=torch.float32)
    alpha[..., 1:7, 1:7] = 1.0
    classes = torch.full((b, k_bank, 1, h, w), float(3), dtype=torch.float32)
    return torch.cat([rgb, alpha, classes], dim=2)


def test_bank_source_forward_has_no_disallowed_breaks() -> None:
    padded = build_fixture(batch_size=2, max_instances=3, image_size=32)
    bank_strategy = BankSource()
    bank_strategy.set_bank_batch(_bank_batch(b=2, h=8, w=8))
    module = BatchCopyPaste(
        BatchCopyPasteConfig(source=BankSourceConfig()),  # pyright: ignore[reportCallIssue]
        source_strategy=bank_strategy,
    )

    reasons = explain_breaks(module, padded)
    allowlist = load_allowlist(ALLOWLIST_PATH)
    _allowed, disallowed = partition(reasons, allowlist)

    assert not disallowed, (
        f"{len(disallowed)} graph-break reason(s) outside "
        f"{ALLOWLIST_PATH.name} on the BankSource forward path: {disallowed}"
    )
