"""Filesystem writer for the gallery directory."""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import shutil
from importlib.metadata import version as _pkg_version
from pathlib import Path

import torch
from pydantic import BaseModel
from torch import Tensor
from torchvision.io import write_png

from segpaste._internal.viz.contact_sheet import compose_contact_sheet
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


def write_gallery(
    out_dir: Path,
    outcomes: list[SampleOutcome],
    *,
    preset: str,
    seed: int,
    batch_size: int,
    device: str,
) -> bool:
    """Write the full artifact tree under *out_dir*. Return True iff all OK."""
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    failed_dir = out_dir / "_failed"

    all_ok = True
    for outcome in outcomes:
        for view, tile in outcome.drilldown.items():
            tile_path = samples_dir / f"{outcome.index:04d}_{view}.png"
            _write_png(tile_path, tile)
            if not outcome.ok:
                if all_ok:
                    failed_dir.mkdir(exist_ok=True)
                _mirror(tile_path, failed_dir / tile_path.name)
        if not outcome.ok:
            all_ok = False

    contact_sheet = compose_contact_sheet([o.drilldown for o in outcomes])
    _write_png(out_dir / "contact_sheet.png", contact_sheet)

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
            source="synthetic",
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


def _write_png(path: Path, tile: Tensor) -> None:
    write_png(tile, str(path))


def _mirror(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _write_json(path: Path, model: BaseModel) -> None:
    payload = model.model_dump(mode="json")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _image_sha256(sample: DenseSample) -> str:
    raw = sample.image.as_subclass(torch.Tensor).contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()
