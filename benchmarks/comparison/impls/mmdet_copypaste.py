"""mmdet :class:`CopyPaste` adapter (ADR-0016 §1).

mmdet is an optional dependency installed via ``[bench-mmdet]``. Import
is lazy; without it the impl reports ``status: "skipped"`` in the
comparison output. mmdet's CopyPaste operates per-sample on NumPy arrays
and reads a ``mix_results`` field on each sample dict pointing at the
paste-source pool — the adapter builds this structure by pairing each
sample with its successor in the batch (cyclic), matching the
"intra-batch source" semantics of the other implementations.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from benchmarks.comparison.impls._base import register
from benchmarks.comparison.workload import CanonicalSample

# mmdet is intentionally not imported at module level; the impl checks
# at instance-construction time and raises NotInstalled, which the sweep
# catches and converts to status=skipped.

_INSTALL_HINT = (
    "mmdet is not installed. Install with `uv sync --group bench-mmdet` "
    "(heavy transitive deps: mmcv, mmengine; may fail on Python 3.13)."
)


class MmdetNotInstalledError(ImportError):
    """Signals to the sweep that mmdet is unavailable on this runner."""


def _import_mmdet() -> dict[str, Any]:
    try:
        from mmdet.datasets.transforms import (  # pyright: ignore[reportMissingImports]
            CopyPaste,
        )
        from mmdet.structures.bbox import (  # pyright: ignore[reportMissingImports]
            HorizontalBoxes,
        )
        from mmdet.structures.mask import (  # pyright: ignore[reportMissingImports]
            BitmapMasks,
        )
    except ImportError as exc:
        raise MmdetNotInstalledError(_INSTALL_HINT) from exc
    return {
        "CopyPaste": CopyPaste,
        "HorizontalBoxes": HorizontalBoxes,
        "BitmapMasks": BitmapMasks,
    }


MmdetBatch = list[dict[str, Any]]


class MmdetImpl:
    """Adapter around ``mmdet.datasets.transforms.CopyPaste``."""

    name = "mmdet"

    def __init__(self) -> None:
        self._transform: Any = None
        self._types: dict[str, Any] | None = None

    def supports_device(self, device: torch.device) -> bool:
        return device.type == "cpu"

    def adapt(self, batches: Sequence[Sequence[CanonicalSample]]) -> list[MmdetBatch]:
        self._types = _import_mmdet()
        self._transform = self._types["CopyPaste"](max_num_pasted=100)
        return [self._pack(batch) for batch in batches]

    def step(self, batch: MmdetBatch) -> object:
        if self._transform is None:
            raise RuntimeError("call adapt() before step()")
        # mmdet's transform mutates / returns a fresh dict per call.
        return [self._transform.transform(sample) for sample in batch]

    def _pack(self, batch: Sequence[CanonicalSample]) -> MmdetBatch:
        assert self._types is not None
        per_sample = [self._to_mmdet_dict(s) for s in batch]
        # Cyclic pairing: sample i's mix_results is [sample (i+1) % B].
        n = len(per_sample)
        return [
            {**per_sample[i], "mix_results": [per_sample[(i + 1) % n]]}
            for i in range(n)
        ]

    def _to_mmdet_dict(self, s: CanonicalSample) -> dict[str, Any]:
        assert self._types is not None
        bitmap_masks_cls = self._types["BitmapMasks"]
        horizontal_boxes_cls = self._types["HorizontalBoxes"]
        # image: [3, H, W] torch -> [H, W, 3] uint8 NumPy
        img = (s.image * 255.0).clamp(0, 255).to(torch.uint8)
        img_hwc = img.permute(1, 2, 0).contiguous().cpu().numpy()
        h, w = int(s.image.shape[-2]), int(s.image.shape[-1])
        masks_np = s.masks.cpu().numpy().astype("uint8")
        return {
            "img": img_hwc,
            "img_shape": (h, w),
            "ori_shape": (h, w),
            "gt_bboxes": horizontal_boxes_cls(s.boxes.to(torch.float32).cpu()),
            "gt_bboxes_labels": s.labels.to(torch.int64).cpu().numpy(),
            "gt_ignore_flags": (s.labels < 0).cpu().numpy(),
            "gt_masks": bitmap_masks_cls(masks_np, height=h, width=w),
        }


register("mmdet", MmdetImpl)
