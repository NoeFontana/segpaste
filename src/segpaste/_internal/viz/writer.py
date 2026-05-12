"""Filesystem writer for the gallery directory.

Writes one ``aug.png`` per sample (consumed as the FO Sample's
``filepath``) plus the three JSON artifacts that ADR-0009 §5 mandates
for paste-into-PR-body.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import platform
from importlib.metadata import version as _pkg_version
from pathlib import Path

import torch
from pydantic import BaseModel
from torch import Tensor
from torchvision.io import write_png

from segpaste._internal.viz.manifest import (
    DatasetManifest,
    DatasetSampleEntry,
    InvariantLog,
    InvariantReportEntry,
    RunManifest,
    SampleInvariantLog,
)
from segpaste._internal.viz.pipeline import SampleOutcome
from segpaste.types import DenseSample


def sample_path(out_dir: Path, index: int) -> Path:
    """Path of the augmented-image PNG for sample *index* under *out_dir*.

    Consumers (``fiftyone_export``) call this helper rather than
    re-constructing the path so renames stay in one place.
    """
    return out_dir / "samples" / f"{index:04d}_aug.png"


def write_gallery(
    out_dir: Path,
    outcomes: list[SampleOutcome],
    *,
    preset: str,
    seed: int,
    batch_size: int,
    device: str,
    source: str,
) -> bool:
    """Write the full artifact tree under *out_dir*. Return True iff all OK."""
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(exist_ok=True)

    all_ok = True
    for outcome in outcomes:
        _write_png(sample_path(out_dir, outcome.index), _to_uint8(outcome.after))
        if not outcome.ok:
            all_ok = False

    _write_json(
        out_dir / "invariant_log.json",
        InvariantLog(
            preset=preset,
            seed=seed,
            samples=tuple(
                SampleInvariantLog(
                    sample_index=o.index,
                    reports=tuple(
                        InvariantReportEntry(
                            name=r.name,
                            ok=r.ok,
                            message=r.message,
                            details=r.details,
                        )
                        for r in o.reports
                    ),
                )
                for o in outcomes
            ),
        ),
    )

    _write_json(
        out_dir / "dataset_manifest.json",
        DatasetManifest(
            source=source,
            sample_count=len(outcomes),
            samples=tuple(
                DatasetSampleEntry(
                    sample_index=o.index,
                    sha256=_image_sha256(o.before),
                    height=int(o.before.image.shape[-2]),
                    width=int(o.before.image.shape[-1]),
                )
                for o in outcomes
            ),
        ),
    )

    _write_json(
        out_dir / "run_manifest.json",
        RunManifest(
            preset=preset,
            seed=seed,
            torch_version=str(torch.__version__),
            segpaste_version=_pkg_version("segpaste"),
            runner=f"{platform.system()}-{platform.machine()}",
            iso_date=_dt.datetime.now(_dt.UTC).date().isoformat(),
            num_samples=len(outcomes),
            batch_size=batch_size,
            device=device,
        ),
    )

    return all_ok


def _to_uint8(sample: DenseSample) -> Tensor:
    """Return the augmented RGB image as a uint8 ``[3, H, W]`` tensor."""
    image = sample.image.as_subclass(torch.Tensor)
    if image.dtype == torch.uint8:
        return image
    return image.clamp(0.0, 1.0).mul(255.0).to(torch.uint8)


def _write_png(path: Path, tile: Tensor) -> None:
    write_png(tile, str(path))


def _write_json(path: Path, model: BaseModel) -> None:
    payload = model.model_dump(mode="json")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _image_sha256(sample: DenseSample) -> str:
    raw = sample.image.as_subclass(torch.Tensor).contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()
