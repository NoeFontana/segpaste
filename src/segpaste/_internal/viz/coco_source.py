"""COCO sample source: HF-cached subset → ``list[DenseSample]`` via LSJ.

The on-disk layout (``images/`` + ``instances_val2017_subset.json``) is
produced by ``scripts/build_eval_subset.py`` and consumed here; the
filename constants below are the single point of agreement.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torchvision.transforms import v2

from segpaste._internal.imports import require_huggingface_hub
from segpaste.augmentation import SanitizeInstances, make_large_scale_jittering
from segpaste.integrations import labels_getter
from segpaste.integrations.coco import CocoDetectionV2, CocoPanopticV2
from segpaste.types import DenseSample

_DEFAULT_HF_CACHE = Path.home() / ".cache" / "segpaste" / "eval-data"
_ANN_FILENAME = "instances_val2017_subset.json"
_IMAGES_DIRNAME = "images"
_PANOPTIC_ANN_FILENAME = "panoptic_val2017.json"
_PANOPTIC_DIRNAME = "panoptic_val2017"


def _coco_transform(image_size: int) -> v2.Transform:
    return v2.Compose(
        [
            v2.ToImage(),
            v2.RandomHorizontalFlip(),
            make_large_scale_jittering(
                output_size=(image_size, image_size),
                min_scale=0.5,
                max_scale=1.5,
            ),
            SanitizeInstances(labels_getter=labels_getter),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )


def snapshot_hf_dataset(repo_id: str, *, cache_dir: Path | None = None) -> Path:
    """Download *repo_id* (HF dataset) to a local cache and return the path."""
    hf_hub = require_huggingface_hub()
    target = cache_dir or _DEFAULT_HF_CACHE
    target.mkdir(parents=True, exist_ok=True)
    local: str = hf_hub.snapshot_download(  # pyright: ignore[reportUnknownMemberType]
        repo_id=repo_id,
        repo_type="dataset",
        cache_dir=str(target),
    )
    return Path(local)


def load_coco_samples(
    coco_dir: Path,
    *,
    count: int,
    image_size: int,
    seed: int,
) -> list[DenseSample]:
    """Load *count* :class:`DenseSample`'s from a COCO subset directory."""
    label_path = coco_dir / _ANN_FILENAME
    if not label_path.is_file():
        raise FileNotFoundError(
            f"COCO annotations not found at {label_path}. "
            "Generate with `scripts/build_eval_subset.py`."
        )
    torch.manual_seed(seed)
    dataset = CocoDetectionV2(
        image_folder=str(coco_dir / _IMAGES_DIRNAME),
        label_path=str(label_path),
        transforms=_coco_transform(image_size),
    )
    available = len(dataset)
    if available < count:
        raise ValueError(
            f"COCO subset has {available} annotated images, asked for {count}"
        )
    return [dataset[i] for i in range(count)]


def load_coco_panoptic_samples(
    coco_dir: Path,
    *,
    count: int,
    image_size: int,
    seed: int,
) -> list[DenseSample]:
    """Load *count* panoptic :class:`DenseSample`'s from a COCO panoptic dir.

    Expects ``panoptic_val2017.json`` + ``panoptic_val2017/*.png`` alongside
    the standard ``images/`` directory. The reviewer-supplied directory must
    follow the upstream COCO panoptic layout (see ADR-0009 §5).
    """
    label_path = coco_dir / _PANOPTIC_ANN_FILENAME
    panoptic_dir = coco_dir / _PANOPTIC_DIRNAME
    hint = (
        "Rebuild the eval subset with "
        "`scripts/build_eval_subset.py --include-panoptic` and re-push."
    )
    if not label_path.is_file():
        raise FileNotFoundError(
            f"COCO panoptic annotations not found at {label_path}. {hint}"
        )
    if not panoptic_dir.is_dir():
        raise FileNotFoundError(
            f"COCO panoptic PNG directory not found at {panoptic_dir}. {hint}"
        )
    torch.manual_seed(seed)
    dataset = CocoPanopticV2(
        image_folder=str(coco_dir / _IMAGES_DIRNAME),
        panoptic_folder=str(panoptic_dir),
        label_path=str(label_path),
        transforms=_coco_transform(image_size),
    )
    available = len(dataset)
    if available < count:
        raise ValueError(
            f"COCO panoptic subset has {available} images, asked for {count}"
        )
    return [dataset[i] for i in range(count)]


def resolve_coco_dir(
    *, hf_repo: str | None, local: Path | None, cache_dir: Path | None = None
) -> Path:
    """Return the directory containing ``images/`` and the COCO JSON.

    Prefers ``local`` when set; otherwise downloads ``hf_repo`` via
    :func:`snapshot_hf_dataset`. Exactly one of the two must be provided.
    """
    if local is not None and hf_repo is not None:
        raise ValueError("Pass exactly one of --coco-local / --coco-hf-repo")
    if local is not None:
        return local
    if hf_repo is None:
        raise ValueError("Pass either --coco-local PATH or --coco-hf-repo REPO")
    return snapshot_hf_dataset(hf_repo, cache_dir=cache_dir)
