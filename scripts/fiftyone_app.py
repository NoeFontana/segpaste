"""Interactive FiftyOne viewer keyed by ``sample_index`` for visual validation.

Drilldown imagery lives under ``out_dir`` (default ``local_gallery/<preset>/``)
and is referenced — not copied — by the FO ``Dataset`` (which lives in
``~/.fiftyone/``). Reviewers can filter the run by ``invariant_passed``,
``failed_checks``, ``K_pasted``, and ``paste_area_frac``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from segpaste._internal.imports import require_fiftyone
from segpaste._internal.viz.fiftyone_export import build_dataset
from segpaste._internal.viz.pipeline import run_preset_batched
from segpaste._internal.viz.synthetic import make_synthetic_samples
from segpaste._internal.viz.writer import write_gallery
from segpaste.augmentation.batch_copy_paste import BatchCopyPasteConfig
from segpaste.presets import get_preset
from segpaste.types import DenseSample

_DEFAULT_SEED = 0xC0FFEE
_DEFAULT_PRESET_LABEL = "default"
_DEFAULT_PORT = 5151
_DEFAULT_COCO_HF_REPO = "NoeFontana/segpaste-eval-data"
_DEFAULT_COCO_IMAGE_SIZE = 512


def _resolve_config(preset: str | None) -> tuple[str, BatchCopyPasteConfig]:
    if preset is None:
        return _DEFAULT_PRESET_LABEL, BatchCopyPasteConfig()
    registered = get_preset(preset)
    return registered.name, registered.batch_copy_paste


def _resolve_out_dir(out_dir: Path | None, preset_label: str) -> Path:
    if out_dir is not None:
        return out_dir
    return Path("local_gallery") / preset_label


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
        choices=["synthetic", "coco"],
        default="synthetic",
        help="Sample source. 'coco' pulls a seeded COCO val2017 subset.",
    )
    parser.add_argument(
        "--task",
        choices=["detection", "panoptic"],
        default="detection",
        help=(
            "COCO task. 'detection' loads instance annotations; 'panoptic' "
            "loads panoptic_*.json + panoptic PNGs. Ignored for synthetic."
        ),
    )
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--coco-hf-repo",
        type=str,
        default=_DEFAULT_COCO_HF_REPO,
        help="HF dataset repo for --source coco. Ignored when --coco-local is set.",
    )
    parser.add_argument(
        "--coco-local",
        type=Path,
        default=None,
        help="Local directory holding images/ + the COCO JSON. Skips HF download.",
    )
    parser.add_argument(
        "--coco-image-size",
        type=int,
        default=_DEFAULT_COCO_IMAGE_SIZE,
        help="LSJ output size (square) for COCO samples.",
    )
    parser.add_argument(
        "--launch",
        dest="launch",
        action="store_true",
        default=True,
        help="Launch the FiftyOne app after building the dataset (default).",
    )
    parser.add_argument(
        "--no-launch",
        dest="launch",
        action="store_false",
        help="Build the dataset and exit; do not open the FiftyOne app.",
    )
    parser.add_argument("--port", type=int, default=_DEFAULT_PORT)
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="FO Dataset name. Default: segpaste-<preset>-<seed:08x>.",
    )
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

    samples: list[DenseSample]
    if args.source == "coco":
        from segpaste._internal.viz.coco_source import (
            load_coco_panoptic_samples,
            load_coco_samples,
            resolve_coco_dir,
        )

        coco_dir = resolve_coco_dir(
            hf_repo=None if args.coco_local else args.coco_hf_repo,
            local=args.coco_local,
        )
        loader = (
            load_coco_panoptic_samples if args.task == "panoptic" else load_coco_samples
        )
        samples = loader(
            coco_dir,
            count=args.num_samples,
            image_size=args.coco_image_size,
            seed=args.seed,
        )
    else:
        samples = make_synthetic_samples(seed=args.seed, count=args.num_samples)
    outcomes = run_preset_batched(
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

    dataset_name = args.dataset_name or f"segpaste-{preset_label}-{args.seed:08x}"
    dataset = build_dataset(
        out_dir=out_dir,
        outcomes=outcomes,
        name=dataset_name,
        info={"preset": preset_label, "seed": args.seed},
    )

    if args.launch:
        fo = require_fiftyone()
        dataset.persistent = True
        dataset.save()
        session = fo.launch_app(dataset, remote=True, port=args.port)
        session.wait(-1)

    if not all_ok:
        print(
            f"fiftyone_app: invariant violations under {out_dir}/_failed",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
