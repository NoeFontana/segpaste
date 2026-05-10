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
from segpaste._internal.gpu.batched_placement import BatchedPlacementConfig
from segpaste.augmentation.batch_copy_paste import BatchCopyPasteConfig

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


@pytest.mark.parametrize(
    "image_size",
    [
        pytest.param(28, id="divisible-snap-only"),
        pytest.param(30, id="non-divisible-pad-and-snap"),
    ],
)
def test_patch_aligned_paste_has_no_disallowed_breaks(image_size: int) -> None:
    """Re-verify compile-clean with ``patch_aligned_paste=True`` (A2).

    Both fixture sizes pin the snap branch (discrete-uniform scale,
    floor-snap translate); ``image_size=30`` additionally fires the
    canvas-pad branch (``pad_canvas_to_multiple``) since 30 % 14 != 0.
    """
    padded = build_fixture(image_size=image_size)
    module = BatchCopyPaste(
        BatchCopyPasteConfig(
            placement=BatchedPlacementConfig(
                scale_range=(0.5, 1.5),
                pad_to_multiple=14,
                patch_aligned_paste=True,
            )
        )
    )

    reasons = explain_breaks(module, padded)
    allowlist = load_allowlist(ALLOWLIST_PATH)
    _allowed, disallowed = partition(reasons, allowlist)

    assert not disallowed, (
        f"{len(disallowed)} graph-break reason(s) outside "
        f"{ALLOWLIST_PATH.name} with patch_aligned_paste=True "
        f"(image_size={image_size}): {disallowed}"
    )
