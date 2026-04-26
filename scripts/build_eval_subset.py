"""Generate a seeded COCO val2017 subset for visual preset evaluation.

Downloads COCO val2017 to ``--coco-root`` (cached), seed-selects
``--num-samples`` images with >=1 instance annotation, writes a portable
subset (``images/`` + filtered ``instances_val2017_subset.json``) under
``--out-dir``, and (unless ``--no-push``) creates/updates the private
HuggingFace dataset repo at ``--hf-repo``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

_VAL_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
_ANNOTATIONS_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)
_PANOPTIC_URL = (
    "http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip"
)
_DEFAULT_NUM = 200
_DEFAULT_SEED = 0xC0FFEE
_DEFAULT_HF_REPO = "NoeFontana/segpaste-eval-data"
_CHUNK_BYTES = 1 << 20


def _download(url: str, dest: Path) -> None:
    if dest.is_file():
        print(f"[cached] {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[download] {url}")
    with (
        urllib.request.urlopen(url) as resp,  # noqa: S310 (trusted COCO host)
        tmp.open("wb") as f,
    ):
        total = int(resp.headers.get("Content-Length", 0))
        read = 0
        while chunk := resp.read(_CHUNK_BYTES):
            f.write(chunk)
            read += len(chunk)
            if total:
                pct = 100 * read / total
                print(
                    f"\r  {read / 1e6:>7.1f} / {total / 1e6:>7.1f} MB ({pct:5.1f}%)",
                    end="",
                    flush=True,
                )
        print()
    tmp.rename(dest)


def _extract(archive: Path, expected: Path, root: Path) -> None:
    if expected.exists():
        print(f"[cached] {expected.name}")
        return
    print(f"[extract] {archive.name}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(root)
    if not expected.exists():
        raise FileNotFoundError(f"{archive} did not produce {expected}")


def _ensure_coco(coco_root: Path) -> tuple[Path, Path]:
    images_dir = coco_root / "val2017"
    ann_path = coco_root / "annotations" / "instances_val2017.json"
    archives = coco_root / "_archives"
    _download(_VAL_IMAGES_URL, archives / "val2017.zip")
    _extract(archives / "val2017.zip", images_dir, coco_root)
    _download(_ANNOTATIONS_URL, archives / "annotations_trainval2017.zip")
    _extract(archives / "annotations_trainval2017.zip", ann_path, coco_root)
    return images_dir, ann_path


def _ensure_coco_panoptic(coco_root: Path) -> tuple[Path, Path]:
    """Ensure panoptic JSON + per-image PNGs are unpacked under *coco_root*.

    The upstream archive nests a second zip — outer extracts the JSONs and
    the inner ``panoptic_val2017.zip`` whose contents (one PNG per image)
    are unpacked into ``annotations/panoptic_val2017/``.
    """
    annotations = coco_root / "annotations"
    pan_json = annotations / "panoptic_val2017.json"
    pan_png_dir = annotations / "panoptic_val2017"
    archives = coco_root / "_archives"
    _download(_PANOPTIC_URL, archives / "panoptic_annotations_trainval2017.zip")
    _extract(archives / "panoptic_annotations_trainval2017.zip", pan_json, coco_root)
    inner = annotations / "panoptic_val2017.zip"
    if not inner.is_file():
        raise FileNotFoundError(
            f"Expected nested panoptic zip at {inner}; outer archive layout changed?"
        )
    _extract(inner, pan_png_dir, annotations)
    return pan_json, pan_png_dir


def _select_image_ids(
    ann_path: Path, num: int, seed: int
) -> tuple[list[int], dict[str, Any]]:
    print(f"[load] {ann_path.name}")
    with ann_path.open() as f:
        data: dict[str, Any] = json.load(f)
    annotated = sorted({a["image_id"] for a in data["annotations"]})
    if len(annotated) < num:
        raise ValueError(
            f"COCO val has {len(annotated)} annotated images, asked for {num}"
        )
    rng = random.Random(seed)
    return sorted(rng.sample(annotated, num)), data


def _filter_and_write(
    *,
    chosen_ids: list[int],
    full_data: dict[str, Any],
    src_images: Path,
    out_dir: Path,
) -> None:
    chosen_set = set(chosen_ids)
    filtered_images = [im for im in full_data["images"] if im["id"] in chosen_set]
    filtered_anns = [a for a in full_data["annotations"] if a["image_id"] in chosen_set]
    subset = {
        "info": full_data.get("info", {}),
        "licenses": full_data.get("licenses", []),
        "categories": full_data["categories"],
        "images": filtered_images,
        "annotations": filtered_anns,
    }

    images_out = out_dir / "images"
    if images_out.exists():
        shutil.rmtree(images_out)
    images_out.mkdir(parents=True)
    print(f"[copy] {len(filtered_images)} images -> {images_out}")
    for im in filtered_images:
        shutil.copy2(src_images / im["file_name"], images_out / im["file_name"])

    ann_out = out_dir / "instances_val2017_subset.json"
    with ann_out.open("w") as f:
        json.dump(subset, f)
    print(f"[wrote] {ann_out.name} ({len(filtered_anns)} annotations)")


def _filter_and_write_panoptic(
    *,
    chosen_ids: list[int],
    pan_json_path: Path,
    pan_png_dir: Path,
    out_dir: Path,
) -> None:
    chosen_set = set(chosen_ids)
    print(f"[load] {pan_json_path.name}")
    with pan_json_path.open() as f:
        full_data: dict[str, Any] = json.load(f)

    filtered_images = [im for im in full_data["images"] if im["id"] in chosen_set]
    filtered_anns = [a for a in full_data["annotations"] if a["image_id"] in chosen_set]
    missing = chosen_set - {a["image_id"] for a in filtered_anns}
    if missing:
        raise ValueError(
            f"{len(missing)} chosen image_ids lack panoptic annotations: "
            f"{sorted(missing)[:5]}..."
        )

    subset = {
        "info": full_data.get("info", {}),
        "licenses": full_data.get("licenses", []),
        "categories": full_data["categories"],
        "images": filtered_images,
        "annotations": filtered_anns,
    }
    ann_out = out_dir / "panoptic_val2017.json"
    with ann_out.open("w") as f:
        json.dump(subset, f)
    print(f"[wrote] {ann_out.name} ({len(filtered_anns)} segments_info entries)")

    png_out = out_dir / "panoptic_val2017"
    if png_out.exists():
        shutil.rmtree(png_out)
    png_out.mkdir(parents=True)
    print(f"[copy] {len(filtered_anns)} panoptic PNGs -> {png_out}")
    for ann in filtered_anns:
        shutil.copy2(pan_png_dir / ann["file_name"], png_out / ann["file_name"])


def _write_readme(
    out_dir: Path, *, num: int, seed: int, fingerprint: str, with_panoptic: bool
) -> None:
    panoptic_layout = (
        "panoptic_val2017.json          # filtered COCO panoptic segments_info\n"
        f"panoptic_val2017/*.png        # {num} PNGs, COCO id-encoded panoptic maps\n"
        if with_panoptic
        else ""
    )
    panoptic_usage = (
        "\nOr the panoptic preset:\n\n"
        "```bash\n"
        "uv run --group viewer --group eval python scripts/fiftyone_app.py \\\n"
        f"    --source coco --task panoptic --num-samples {num} \\\n"
        "    --preset coco-panoptic\n"
        "```\n"
        if with_panoptic
        else ""
    )
    body = f"""---
license: cc-by-4.0
task_categories:
  - object-detection
  - image-segmentation
size_categories:
  - n<1K
---

# segpaste-eval-data

A {num}-image seeded subset of COCO val2017, used by
[segpaste](https://github.com/NoeFontana/segpaste)'s
`scripts/fiftyone_app.py --source coco` for visual evaluation of
`BatchCopyPaste` presets.

## Provenance

- Source: COCO val2017 (https://cocodataset.org), CC-BY-4.0
- Selection seed: `0x{seed:08X}`
- Selection criterion: image has >=1 instance annotation; subset is
  `sorted(random.Random(seed).sample(annotated_image_ids, {num}))`
- Annotations: filtered `instances_val2017.json`, all 80 categories retained
- Subset fingerprint (sha256 of sorted image_id list): `{fingerprint}`

## Layout

```
images/*.jpg                  # {num} JPEGs, original COCO filenames
instances_val2017_subset.json # COCO-format annotations for the subset
{panoptic_layout}```

## Usage

```python
from segpaste.integrations import CocoDetectionV2
ds = CocoDetectionV2(
    image_folder="images",
    label_path="instances_val2017_subset.json",
)
```

Or via the visual viewer (no manual download):

```bash
uv run --group viewer --group eval python scripts/fiftyone_app.py \\
    --source coco --num-samples {num} --preset <name>
```
{panoptic_usage}"""
    (out_dir / "README.md").write_text(body)
    print("[wrote] README.md")


def _fingerprint(chosen_ids: list[int]) -> str:
    payload = ",".join(str(i) for i in chosen_ids).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _push_to_hf(out_dir: Path, repo_id: str, *, private: bool) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    print(f"[hf] create_repo {repo_id} private={private}")
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    print(f"[hf] upload_folder {out_dir.name} -> {repo_id}")
    api.upload_folder(
        folder_path=str(out_dir),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Generate seeded COCO val2017 subset",
    )
    print(f"[hf] done -> https://huggingface.co/datasets/{repo_id}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coco-root", type=Path, default=Path.home() / "dataset" / "coco"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path.home() / "dataset" / "segpaste-eval-data",
    )
    parser.add_argument("--num-samples", type=int, default=_DEFAULT_NUM)
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    parser.add_argument("--hf-repo", type=str, default=_DEFAULT_HF_REPO)
    parser.add_argument(
        "--public",
        action="store_true",
        help="Push as a public repo (default: private).",
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Build the local subset and exit; do not push to HF.",
    )
    parser.add_argument(
        "--include-panoptic",
        action="store_true",
        help=(
            "Also fetch panoptic_annotations_trainval2017.zip and emit "
            "panoptic_val2017.json + panoptic_val2017/*.png alongside the "
            "instance subset (~800MB additional download)."
        ),
    )
    args = parser.parse_args(argv)

    if args.num_samples <= 0:
        print(
            f"--num-samples must be positive, got {args.num_samples}",
            file=sys.stderr,
        )
        return 2

    coco_images, coco_ann = _ensure_coco(args.coco_root)
    chosen, full_data = _select_image_ids(coco_ann, args.num_samples, args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _filter_and_write(
        chosen_ids=chosen,
        full_data=full_data,
        src_images=coco_images,
        out_dir=args.out_dir,
    )
    if args.include_panoptic:
        pan_json, pan_png_dir = _ensure_coco_panoptic(args.coco_root)
        _filter_and_write_panoptic(
            chosen_ids=chosen,
            pan_json_path=pan_json,
            pan_png_dir=pan_png_dir,
            out_dir=args.out_dir,
        )
    _write_readme(
        args.out_dir,
        num=args.num_samples,
        seed=args.seed,
        fingerprint=_fingerprint(chosen),
        with_panoptic=args.include_panoptic,
    )

    if args.no_push:
        print("[skip] --no-push set, not pushing to HF")
        return 0
    _push_to_hf(args.out_dir, args.hf_repo, private=not args.public)
    return 0


if __name__ == "__main__":
    sys.exit(main())
