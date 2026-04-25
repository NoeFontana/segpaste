"""Compile-clean gate for :class:`BatchCopyPaste.forward` (ADR-0008 §D7)."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.compile_explain import (
    build_fixture,
    explain_breaks,
    load_allowlist,
    partition,
)
from segpaste import BatchCopyPaste

pytest.importorskip("torch._dynamo")

ALLOWLIST_PATH = Path(__file__).parents[1] / "scripts" / "compile_allowlist.txt"


def test_forward_has_no_disallowed_breaks() -> None:
    """``torch._dynamo.explain`` reports zero breaks outside the allow-list.

    Pins ADR-0008 §D7's compile-clean invariant. Additions to the allow-list
    require an ADR amendment.
    """
    padded = build_fixture()
    module = BatchCopyPaste()

    reasons = explain_breaks(module, padded)
    allowlist = load_allowlist(ALLOWLIST_PATH)
    _allowed, disallowed = partition(reasons, allowlist)

    assert not disallowed, (
        f"{len(disallowed)} graph-break reason(s) outside "
        f"{ALLOWLIST_PATH.name}: {disallowed}"
    )
