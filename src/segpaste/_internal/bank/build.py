"""Build path for instance banks (ADR-0011 PR5).

Two stages:

1. ``crops_from_coco`` reads COCO instance JSON, walks
   ``(image_id, annotation_id)`` in lexicographic order, and yields
   ``BankCrop`` records cropped to the annotation bbox + padded to the
   target ``(h, w)``. Deterministic by construction — no shuffling.
2. ``build_bank`` packs an iterable of ``BankCrop`` into the requested
   on-disk format (``memmap`` or ``lmdb``) and stamps ``meta.json``.

The script ``scripts/build_instance_bank.py`` glues these two together.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Literal

from segpaste._internal.bank.lmdb_backend import write_lmdb_bank
from segpaste._internal.bank.memmap import write_memmap_bank
from segpaste._internal.bank.protocol import BankCrop
from segpaste._internal.bank.webdataset_backend import write_webdataset_bank
from segpaste._internal.imports import require_numpy

Format = Literal["memmap", "lmdb", "webdataset"]


def crops_from_coco(
    coco_json: Path,
    image_dir: Path,
    *,
    crop_size: tuple[int, int],
    min_area: int = 256,
) -> Iterator[BankCrop]:
    """Yield ``BankCrop`` records from a COCO instance dataset.

    Iteration order is lexicographic on ``(image_id, annotation_id)`` so
    two builds with the same inputs produce byte-identical banks.
    Annotations with bbox area below ``min_area`` are skipped — they
    rarely contain enough signal to reward the bank slot.
    """
    np = require_numpy()
    h, w = crop_size

    # Lazy imports keep ``faster_coco_eval`` (project-wide already) and PIL
    # off the import-time module graph for environments that build banks
    # offline only.
    from faster_coco_eval import COCO  # type: ignore[import-untyped]
    from PIL import Image as PILImage  # type: ignore[import-untyped]

    coco = COCO(str(coco_json))
    image_dir = Path(image_dir)
    sorted_image_ids = sorted(int(x) for x in coco.imgs)
    for image_id in sorted_image_ids:
        info = coco.imgs[image_id]
        ann_ids = coco.getAnnIds(imgIds=[image_id])
        if not ann_ids:
            continue
        with PILImage.open(image_dir / info["file_name"]).convert("RGB") as pil_img:
            np_img = np.asarray(pil_img, dtype=np.uint8)  # H, W, 3
        for ann_id in sorted(int(x) for x in ann_ids):
            ann = coco.anns[ann_id]
            if ann.get("iscrowd"):
                continue
            x, y, bw, bh = ann["bbox"]
            if bw * bh < min_area:
                continue
            x = round(x)
            y = round(y)
            bw = round(bw)
            bh = round(bh)
            x2 = min(x + bw, np_img.shape[1])
            y2 = min(y + bh, np_img.shape[0])
            x = max(x, 0)
            y = max(y, 0)
            if x2 <= x or y2 <= y:
                continue

            patch = np_img[y:y2, x:x2]  # ph, pw, 3
            mask_full = coco.annToMask(ann).astype(np.bool_)  # H, W
            mask_patch = mask_full[y:y2, x:x2]

            # Center-pad to (h, w). Crops larger than the target are
            # center-cropped to fit — same dimension for image and mask.
            patch_chw = np.transpose(patch, (2, 0, 1))  # 3, ph, pw
            image_buf = np.zeros((3, h, w), dtype=np.uint8)
            alpha_buf = np.zeros((1, h, w), dtype=np.bool_)
            ph, pw = patch_chw.shape[1:]
            sh = min(ph, h)
            sw = min(pw, w)
            src_y = (ph - sh) // 2
            src_x = (pw - sw) // 2
            dst_y = (h - sh) // 2
            dst_x = (w - sw) // 2
            image_buf[:, dst_y : dst_y + sh, dst_x : dst_x + sw] = patch_chw[
                :, src_y : src_y + sh, src_x : src_x + sw
            ]
            alpha_buf[0, dst_y : dst_y + sh, dst_x : dst_x + sw] = mask_patch[
                src_y : src_y + sh, src_x : src_x + sw
            ]

            yield BankCrop(
                image=image_buf,  # type: ignore[arg-type]
                alpha=alpha_buf,  # type: ignore[arg-type]
                class_id=int(ann["category_id"]),
                embedding=None,
            )


def build_bank(
    crops: Iterable[BankCrop],
    *,
    out_path: Path,
    out_format: Format,
    num_classes: int,
    crop_size: tuple[int, int],
    base_seed: int = 0,
    segpaste_version: str = "0",
) -> Path:
    """Materialize an iterable of crops into an on-disk bank.

    Buffers the iterable into numpy arrays before writing — fine at the
    crop counts banks reach in practice (≤ low millions). For huge
    builds, prefer the streaming write path (LMDBBank can take an
    iterator directly; future optimization).
    """
    np = require_numpy()
    h, w = crop_size

    images: list[Any] = []
    alphas: list[Any] = []
    classes: list[int] = []
    embeddings: list[Any] = []
    has_embeddings: bool | None = None

    for crop in crops:
        # ``BankCrop.image`` is typed as ``torch.Tensor`` in the protocol but
        # the COCO crop iterator yields raw numpy ndarrays for cheap stack
        # ops below. Both are accepted here; we only require ``shape``.
        img_arr = np.asarray(crop.image, dtype=np.uint8)
        alpha_arr = np.asarray(crop.alpha, dtype=np.bool_)
        if img_arr.shape != (3, h, w):
            raise ValueError(f"crop image shape {img_arr.shape} != (3, {h}, {w})")
        if alpha_arr.shape != (1, h, w):
            raise ValueError(f"crop alpha shape {alpha_arr.shape} != (1, {h}, {w})")
        images.append(img_arr)
        alphas.append(alpha_arr)
        classes.append(int(crop.class_id))
        crop_has_emb = crop.embedding is not None
        if has_embeddings is None:
            has_embeddings = crop_has_emb
        elif has_embeddings != crop_has_emb:
            raise ValueError("inconsistent embedding presence across crops")
        if crop_has_emb:
            embeddings.append(np.asarray(crop.embedding, dtype=np.float16))

    if not images:
        raise ValueError("no crops to write — empty iterable")

    images_arr = np.stack(images, axis=0)
    alphas_arr = np.stack(alphas, axis=0)
    classes_arr = np.asarray(classes, dtype=np.int64)
    embeddings_arr = (
        np.stack(embeddings, axis=0) if has_embeddings and embeddings else None
    )

    if out_format == "memmap":
        return write_memmap_bank(
            out_path,
            images=images_arr,
            alpha=alphas_arr,
            classes=classes_arr,
            num_classes=num_classes,
            embeddings=embeddings_arr,
            build_seed=base_seed,
            segpaste_version=segpaste_version,
        )
    if out_format == "lmdb":
        return write_lmdb_bank(
            out_path,
            images=images_arr,
            alpha=alphas_arr,
            classes=classes_arr,
            num_classes=num_classes,
            embeddings=embeddings_arr,
            build_seed=base_seed,
            segpaste_version=segpaste_version,
        )
    if out_format == "webdataset":
        return write_webdataset_bank(
            out_path,
            images=images_arr,
            alpha=alphas_arr,
            classes=classes_arr,
            num_classes=num_classes,
            embeddings=embeddings_arr,
            build_seed=base_seed,
            segpaste_version=segpaste_version,
        )
    raise ValueError(f"unknown out_format: {out_format!r}")


def write_provenance(out_path: Path, provenance: dict[str, Any]) -> None:
    """Drop a ``provenance.json`` alongside the bank for downstream auditing."""
    out_path = Path(out_path)
    with (out_path / "provenance.json").open("w") as fh:
        json.dump(provenance, fh, sort_keys=True, indent=2)


__all__ = ["Format", "build_bank", "crops_from_coco", "write_provenance"]
