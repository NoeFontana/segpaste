"""Local visualization renderer for `BatchCopyPaste` (ADR-0009 §5).

Runs a preset's :class:`BatchCopyPasteConfig` over a sample source, renders
per-sample drilldowns + a contact sheet, and emits structured JSON for paste
into a PR body. Any invariant violation populates ``_failed/`` and exits
non-zero.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import torch

from segpaste._internal.viz.pipeline import SampleOutcome, run_preset
from segpaste._internal.viz.synthetic import make_synthetic_samples
from segpaste._internal.viz.writer import write_gallery
from segpaste.augmentation.batch_copy_paste import BatchCopyPasteConfig
from segpaste.presets import get_preset
from segpaste.types import DenseSample

_DEFAULT_SEED = 0xC0FFEE
_DEFAULT_PRESET_LABEL = "default"


def _resolve_config(preset: str | None) -> tuple[str, BatchCopyPasteConfig]:
    if preset is None:
        return _DEFAULT_PRESET_LABEL, BatchCopyPasteConfig()
    registered = get_preset(preset)
    return registered.name, registered.batch_copy_paste


def _resolve_out_dir(out_dir: Path | None, preset_label: str) -> Path:
    if out_dir is not None:
        return out_dir
    return Path("local_gallery") / preset_label


def _run(
    config: BatchCopyPasteConfig,
    samples: list[DenseSample],
    *,
    seed: int,
    batch_size: int,
    device: torch.device,
) -> list[SampleOutcome]:
    outcomes: list[SampleOutcome] = []
    for start in range(0, len(samples), batch_size):
        chunk = samples[start : start + batch_size]
        chunk_out = run_preset(
            config, chunk, seed=seed + start, device=device, force_overlap=False
        )
        outcomes.extend(replace(o, index=start + o.index) for o in chunk_out)
    return outcomes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Registered preset name. Omit to use BatchCopyPasteConfig() defaults.",
    )
    parser.add_argument(
        "--source",
        choices=["synthetic"],
        default="synthetic",
        help="Sample source.",
    )
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args(argv)

    preset_label, config = _resolve_config(args.preset)
    out_dir = _resolve_out_dir(args.out_dir, preset_label)
    device = torch.device(args.device)

    if args.num_samples <= 0:
        print(
            f"--num-samples must be positive, got {args.num_samples}", file=sys.stderr
        )
        return 2
    if args.batch_size <= 0:
        print(f"--batch-size must be positive, got {args.batch_size}", file=sys.stderr)
        return 2

    samples = make_synthetic_samples(seed=args.seed, count=args.num_samples)
    outcomes = _run(
        config,
        samples,
        seed=args.seed,
        batch_size=args.batch_size,
        device=device,
    )

    all_ok = write_gallery(
        out_dir,
        outcomes,
        preset=preset_label,
        seed=args.seed,
        batch_size=args.batch_size,
        device=str(device),
    )

    if not all_ok:
        print(
            f"visualize_preset: invariant violations under {out_dir}/_failed",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
