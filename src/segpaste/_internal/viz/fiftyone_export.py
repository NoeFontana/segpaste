"""Build a FiftyOne ``Dataset`` keyed by ``sample_index`` from outcomes."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from segpaste._internal.imports import require_fiftyone
from segpaste._internal.viz.paste_stats import compute_paste_stats
from segpaste._internal.viz.pipeline import SampleOutcome
from segpaste._internal.viz.writer import sample_path
from segpaste.types import DenseSample

if TYPE_CHECKING:
    import fiftyone as fo


def _to_detections(sample: DenseSample) -> fo.Detections:
    """Convert a :class:`DenseSample`'s instance fields to ``fo.Detections``.

    Each detection carries a normalized ``[x, y, w, h]`` box and a 2D
    binary mask cropped to the bbox extent — the FO-native instance
    segmentation representation. Boxes are clamped to the image rect
    before cropping so any sub-pixel drift past the canvas does not
    produce a zero-width slice.
    """
    fo = require_fiftyone()

    if (
        sample.instance_masks is None
        or sample.instance_masks.shape[0] == 0
        or sample.boxes.shape[0] == 0
    ):
        return fo.Detections(detections=[])

    h, w = sample.image.shape[-2:]
    masks = sample.instance_masks.as_subclass(torch.Tensor).to(torch.bool).cpu().numpy()
    boxes = sample.boxes.as_subclass(torch.Tensor).cpu().numpy()
    labels = sample.labels.cpu().tolist() if sample.labels.numel() else []

    detections: list[fo.Detection] = []
    for i in range(masks.shape[0]):
        x1, y1, x2, y2 = (float(v) for v in boxes[i])
        x1c = max(0.0, min(float(w), x1))
        y1c = max(0.0, min(float(h), y1))
        x2c = max(0.0, min(float(w), x2))
        y2c = max(0.0, min(float(h), y2))
        if x2c <= x1c or y2c <= y1c:
            continue
        x1i, y1i, x2i, y2i = int(x1c), int(y1c), round(x2c), round(y2c)
        if x2i <= x1i or y2i <= y1i:
            continue
        cropped = masks[i, y1i:y2i, x1i:x2i].astype("uint8")
        label = str(int(labels[i])) if i < len(labels) else "instance"
        detections.append(
            fo.Detection(
                label=label,
                bounding_box=[x1c / w, y1c / h, (x2c - x1c) / w, (y2c - y1c) / h],
                mask=cropped,
            )
        )
    return fo.Detections(detections=detections)


def build_dataset(
    *,
    out_dir: Path,
    outcomes: Sequence[SampleOutcome],
    name: str | None = None,
    info: dict[str, Any] | None = None,
) -> fo.Dataset:
    """Return a FO ``Dataset`` keyed by ``sample_index``.

    The ``aug`` drilldown tile (raw augmented image) is the primary
    ``filepath``; instance masks/boxes are attached to the FO sample as
    a native ``detections`` field so FiftyOne renders the overlay through
    its UI rather than baking pixels. ``orig`` and ``overlay`` (diff)
    paths are exposed as ``original_filepath`` / ``overlay_filepath``,
    and the pre-augmentation instance fields are mirrored as
    ``original_detections``. Per-sample invariant outcomes and paste
    stats are populated as filterable fields. *info* is assigned to
    ``dataset.info`` verbatim; persistence (``dataset.save()``) is the
    caller's job.
    """
    fo = require_fiftyone()

    dataset = fo.Dataset(name=name, overwrite=name is not None)

    fo_samples: list[Any] = []
    for outcome in outcomes:
        kwargs: dict[str, Any] = {
            "filepath": str(sample_path(out_dir, outcome.index, "aug")),
            "original_filepath": str(sample_path(out_dir, outcome.index, "orig")),
            "overlay_filepath": str(sample_path(out_dir, outcome.index, "overlay")),
            "sample_index": outcome.index,
            "invariant_passed": outcome.ok,
            "failed_checks": [r.name for r in outcome.reports if not r.ok],
            "detections": _to_detections(outcome.after),
            "original_detections": _to_detections(outcome.before),
        }

        stats = compute_paste_stats(outcome.before, outcome.after)
        if stats is not None:
            kwargs["K_pasted"] = stats.K_pasted
            kwargs["paste_area_frac"] = stats.paste_area_frac

        fo_samples.append(fo.Sample(**kwargs))

    dataset.add_samples(fo_samples)

    if info:
        dataset.info = info

    return dataset
