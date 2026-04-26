"""Per-sample paste statistics derived from a ``(before, after)`` pair."""

from __future__ import annotations

from dataclasses import dataclass

from segpaste.types import DenseSample


@dataclass(frozen=True, slots=True)
class PasteStats:
    """Summary stats for the pasted instances in an augmented sample."""

    K_pasted: int
    paste_area_frac: float


def compute_paste_stats(before: DenseSample, after: DenseSample) -> PasteStats | None:
    """Return paste stats, or ``None`` if INSTANCE modality is absent on either side."""
    if (
        before.instance_ids is None
        or after.instance_ids is None
        or after.instance_masks is None
    ):
        return None

    before_ids = set(before.instance_ids.tolist())
    after_ids_list = after.instance_ids.tolist()
    pasted_ids = set(after_ids_list) - before_ids
    k_pasted = len(pasted_ids)

    if k_pasted == 0:
        return PasteStats(K_pasted=0, paste_area_frac=0.0)

    pasted_idx = [i for i, iid in enumerate(after_ids_list) if iid in pasted_ids]
    masks = after.instance_masks[pasted_idx]
    pasted_area = int(masks.any(dim=0).sum().item())
    h, w = after.image.shape[-2:]
    return PasteStats(K_pasted=k_pasted, paste_area_frac=pasted_area / float(h * w))
