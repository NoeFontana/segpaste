#!/usr/bin/env python3
"""Build a segpaste instance bank from a COCO dataset (ADR-0011 PR5).

Walks ``--coco-json`` in ``(image_id, annotation_id)`` order, crops
each annotation to its bbox + center-pads to ``--crop-size``, and
writes a memmap or lmdb bank under ``--out-path``. The build is
deterministic — two runs with the same inputs produce byte-identical
banks (verified by ``meta.json#sha256``).

Usage::

    uv run python scripts/build_instance_bank.py \
        --coco-json annotations/instances_train.json \
        --image-dir train2017/ \
        --out-format memmap \
        --out-path banks/coco_train_v1/ \
        --crop-size 224 \
        --min-area 256 \
        --num-classes 80
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast

from segpaste._internal.bank.build import (
    Format,
    build_bank,
    crops_from_coco,
    write_provenance,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-json", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument(
        "--out-format", choices=("memmap", "lmdb", "webdataset"), required=True
    )
    parser.add_argument("--out-path", type=Path, required=True)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--min-area", type=int, default=256)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument(
        "--segpaste-version",
        type=str,
        default="0",
        help="Stamp written into meta.json for cache-key bookkeeping.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    crop_size = (args.crop_size, args.crop_size)
    crops = crops_from_coco(
        coco_json=args.coco_json,
        image_dir=args.image_dir,
        crop_size=crop_size,
        min_area=args.min_area,
    )
    out_format = cast(Format, args.out_format)
    print(
        f"[build_instance_bank] reading {args.coco_json}, "
        f"writing {out_format} bank to {args.out_path}"
    )
    out_path = build_bank(
        crops,
        out_path=args.out_path,
        out_format=out_format,
        num_classes=args.num_classes,
        crop_size=crop_size,
        base_seed=args.base_seed,
        segpaste_version=args.segpaste_version,
    )
    write_provenance(
        out_path,
        {
            "coco_json": str(args.coco_json),
            "image_dir": str(args.image_dir),
            "crop_size": list(crop_size),
            "min_area": int(args.min_area),
            "num_classes": int(args.num_classes),
            "base_seed": int(args.base_seed),
            "segpaste_version": str(args.segpaste_version),
        },
    )
    print(f"[build_instance_bank] done at {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
