"""Contact-sheet composition via ``torchvision.utils.make_grid``.

Each sample contributes one row of three tiles (orig, aug, overlay).
The grid is rendered at ``nrow=3`` so each output column is one tile
type — the visual scan reads vertically by sample, horizontally by
view.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torchvision.utils import make_grid

_VIEW_ORDER: tuple[str, ...] = ("orig", "aug", "overlay")


def compose_contact_sheet(
    drilldowns: list[dict[str, Tensor]], padding: int = 2
) -> Tensor:
    """Stack per-sample drilldown dicts into a single uint8 grid tensor.

    Each input dict carries the three tiles produced by
    ``render_drilldown``. Output is a ``[3, H', W']`` uint8 tensor.
    """
    if not drilldowns:
        raise ValueError("drilldowns must be non-empty")

    tiles = [d[view] for d in drilldowns for view in _VIEW_ORDER]
    return make_grid(torch.stack(tiles), nrow=len(_VIEW_ORDER), padding=padding)
