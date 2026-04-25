"""KS soft-report for :class:`BatchCopyPaste` vs. pre-deletion CPU outputs.

ADR-0008 C9 / §statistics. The 30-day burn-in (per D6) records distances
as a CI artifact without asserting. After the burn-in closes, ADR-0008 is
amended with a threshold and the final ``assert`` is added here.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import torch
from torchvision import tv_tensors

from segpaste import BatchCopyPaste, PaddedBatchedDenseSample
from segpaste.types import (
    BatchedDenseSample,
    DenseSample,
    InstanceMask,
)

SNAPSHOT_PATH = Path(__file__).parent / "fixtures" / "ks_snapshot.pt"
ARTIFACT_DIR = Path(os.environ.get("KS_ARTIFACT_DIR", "ks_report"))
N_SAMPLES = 200
H = W = 64
NUM_CLASSES = 20


def _sample(seed: int) -> DenseSample:
    gen = torch.Generator().manual_seed(seed * 2654435761 & 0xFFFFFFFF)
    n = int(torch.randint(2, 5, (1,), generator=gen).item())
    image = tv_tensors.Image(torch.rand(3, H, W, generator=gen))
    masks = torch.zeros(n, H, W, dtype=torch.bool)
    boxes: list[list[int]] = []
    for i in range(n):
        side = int(torch.randint(8, 20, (1,), generator=gen).item())
        x1 = int(torch.randint(0, W - side, (1,), generator=gen).item())
        y1 = int(torch.randint(0, H - side, (1,), generator=gen).item())
        masks[i, y1 : y1 + side, x1 : x1 + side] = True
        boxes.append([x1, y1, x1 + side, y1 + side])
    return DenseSample(
        image=image,
        boxes=tv_tensors.BoundingBoxes(  # pyright: ignore[reportCallIssue]
            torch.tensor(boxes, dtype=torch.float32),
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=(H, W),
        ),
        labels=torch.randint(1, NUM_CLASSES, (n,), generator=gen, dtype=torch.int64),
        instance_ids=torch.arange(n, dtype=torch.int32),
        instance_masks=InstanceMask(masks),
    )


def _padded_pair(seed: int) -> PaddedBatchedDenseSample:
    samples = [_sample(seed * 2 + i) for i in range(2)]
    return BatchedDenseSample.from_samples(samples).to_padded(max_instances=8)


def _ks_statistic(a: torch.Tensor, b: torch.Tensor) -> float:
    """Two-sample Kolmogorov-Smirnov distance ``sup|F_a - F_b|``."""
    if a.numel() == 0 or b.numel() == 0:
        return float("nan")
    a_sorted = a.double().sort().values
    b_sorted = b.double().sort().values
    eval_pts = torch.cat([a_sorted, b_sorted])
    ca = torch.searchsorted(a_sorted, eval_pts, right=True) / a.numel()
    cb = torch.searchsorted(b_sorted, eval_pts, right=True) / b.numel()
    return float((ca - cb).abs().max().item())


def _seeded(i: int) -> torch.Generator:
    return torch.Generator().manual_seed(i * 2654435761 & 0xFFFFFFFF)


def _collect_histograms(module: BatchCopyPaste) -> dict[str, torch.Tensor]:
    paste_area: list[float] = []
    num_new: list[int] = []
    class_counts: list[torch.Tensor] = []

    # _padded_pair uses batch_size=2, so N_SAMPLES // 2 outer iterations
    # produce exactly N_SAMPLES histogram entries — matching ks_snapshot.pt.
    for i in range(N_SAMPLES // 2):
        padded = _padded_pair(seed=i)
        placement = module.placement_sampler(padded, _seeded(i))
        warped = module.propagator(padded, placement)
        paste_gate = placement.paste_valid.view(
            placement.paste_valid.shape[0], placement.paste_valid.shape[1], 1, 1
        )
        assert warped.instance_masks is not None
        pm = (warped.instance_masks & paste_gate).any(dim=1)

        merged_labels = torch.where(placement.paste_valid, warped.labels, padded.labels)
        merged_valid = padded.instance_valid | placement.paste_valid

        for b in range(padded.batch_size):
            paste_area.append(float(pm[b].float().mean().item()))
            num_new.append(int(placement.paste_valid[b].sum().item()))
            labels_b = merged_labels[b][merged_valid[b]]
            class_counts.append(torch.bincount(labels_b, minlength=NUM_CLASSES))

    return {
        "paste_area": torch.tensor(paste_area, dtype=torch.float64),
        "num_new_instances": torch.tensor(num_new, dtype=torch.int64),
        "class_counts": torch.stack(class_counts, dim=0),
    }


def test_ks_soft_report() -> None:
    """Record KS distances per modality/histogram pair as a CI artifact.

    Soft-gate per ADR-0008 §D6: no assertion during the 30-day burn-in.
    The dashboard consumes the JSON artifact emitted here.
    """
    snapshot = torch.load(SNAPSHOT_PATH, weights_only=False)
    module = BatchCopyPaste()
    current = _collect_histograms(module)

    report: dict[str, dict[str, float]] = {}
    for wrapper, hist in snapshot.items():
        wrapper_report: dict[str, float] = {}
        for field in ("paste_area", "num_new_instances"):
            wrapper_report[field] = _ks_statistic(hist[field], current[field])
        max_per_class: float = 0.0
        for c in range(NUM_CLASSES):
            d = _ks_statistic(hist["class_counts"][:, c], current["class_counts"][:, c])
            if not math.isnan(d) and d > max_per_class:
                max_per_class = d
        wrapper_report["class_counts_max"] = max_per_class
        report[wrapper] = wrapper_report

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    artifact_path = ARTIFACT_DIR / "ks_distances.json"
    artifact_path.write_text(
        json.dumps({"schema_version": 1, "n": N_SAMPLES, "distances": report}, indent=2)
    )

    for wrapper, fields in report.items():
        for name, d in fields.items():
            print(f"KS[{wrapper}.{name}] = {d:.4f}")
