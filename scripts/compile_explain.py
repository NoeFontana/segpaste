"""Compile-clean gate for :class:`BatchCopyPaste.forward` (ADR-0008 §D7).

Runs ``torch._dynamo.explain`` on a fixture :class:`PaddedBatchedDenseSample`
and fails on any graph-break reason not present in the allow-list.

The allow-list starts empty at M4; additions require an ADR-0008 amendment.

Exit codes
----------
* ``0`` — every break reason is covered by the allow-list.
* ``1`` — one or more reasons fall outside the allow-list.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, cast

import torch
import torch._dynamo
from torchvision import tv_tensors

from segpaste import BatchCopyPaste, PaddedBatchedDenseSample
from segpaste.types import BatchedDenseSample, DenseSample, InstanceMask


def _sample(seed: int, h: int, w: int, k: int) -> DenseSample:
    gen = torch.Generator().manual_seed(seed)
    masks = torch.zeros(k, h, w, dtype=torch.bool)
    masks[:, h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = True
    return DenseSample(
        image=tv_tensors.Image(torch.rand(3, h, w, generator=gen)),
        boxes=tv_tensors.BoundingBoxes(  # pyright: ignore[reportCallIssue]
            torch.tensor(
                [[w // 4, h // 4, 3 * w // 4, 3 * h // 4]] * k, dtype=torch.float32
            ),
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=(h, w),
        ),
        labels=torch.arange(1, k + 1, dtype=torch.int64),
        instance_ids=torch.arange(k, dtype=torch.int32),
        instance_masks=InstanceMask(masks),
    )


def build_fixture(
    batch_size: int = 2,
    max_instances: int = 3,
    image_size: int = 32,
) -> PaddedBatchedDenseSample:
    """Minimal padded batch that exercises the canonical entry path.

    Scope (batch/K/HW) does not change which graph-break reasons dynamo
    surfaces, so a small fixture keeps the explain trace fast.
    """
    samples = [
        _sample(seed=i, h=image_size, w=image_size, k=max_instances)
        for i in range(batch_size)
    ]
    return BatchedDenseSample.from_samples(samples).to_padded(
        max_instances=max_instances
    )


def load_allowlist(path: Path) -> list[str]:
    if not path.exists():
        return []
    entries: list[str] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        entries.append(stripped)
    return entries


def explain_breaks(
    module: BatchCopyPaste, padded: PaddedBatchedDenseSample
) -> list[str]:
    torch._dynamo.reset()  # pyright: ignore[reportPrivateUsage]
    explainer = cast(Any, torch._dynamo.explain(module.forward))
    result = explainer(padded)
    return [
        str(br.reason)
        for br in result.break_reasons
        if bool(getattr(br, "graph_break", True))
    ]


def partition(reasons: list[str], allowlist: list[str]) -> tuple[list[str], list[str]]:
    allowed: list[str] = []
    disallowed: list[str] = []
    for reason in reasons:
        if any(entry in reason for entry in allowlist):
            allowed.append(reason)
        else:
            disallowed.append(reason)
    return allowed, disallowed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allowlist",
        type=Path,
        default=Path(__file__).with_name("compile_allowlist.txt"),
    )
    args = parser.parse_args(argv)

    padded = build_fixture()
    module = BatchCopyPaste()
    reasons = explain_breaks(module, padded)

    allowlist = load_allowlist(args.allowlist)
    allowed, disallowed = partition(reasons, allowlist)

    print(f"graph breaks: {len(reasons)} total, {len(allowed)} allow-listed")
    for reason in allowed:
        print(f"  [OK] {reason}")
    for reason in disallowed:
        print(f"  [FAIL] {reason}", file=sys.stderr)

    if disallowed:
        print(
            f"\n{len(disallowed)} graph-break reason(s) are not in the "
            f"allow-list at {args.allowlist}. Per ADR-0008 §D7, any new "
            "entry requires an ADR amendment.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
