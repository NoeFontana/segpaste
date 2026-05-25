"""Implementation adapters for the comparison harness (ADR-0016 §1).

Each module here registers a single :class:`Implementation` factory in
the :data:`REGISTRY` dict in :mod:`._base`. The :mod:`..sweep` CLI
iterates ``REGISTRY.items()`` to drive the grid.
"""

from __future__ import annotations

from benchmarks.comparison.impls._base import (
    Implementation,
    ImplementationFactory,
    InputBatch,
    register,
    registry,
)

__all__ = [
    "Implementation",
    "ImplementationFactory",
    "InputBatch",
    "register",
    "registry",
]

# Side-effect imports populate the registry — must follow the helpers above.
import benchmarks.comparison.impls.mmdet_copypaste  # pyright: ignore[reportUnusedImport]
import benchmarks.comparison.impls.segpaste_batchcopypaste  # pyright: ignore[reportUnusedImport]
import benchmarks.comparison.impls.torchvision_simple  # noqa: F401  # pyright: ignore[reportUnusedImport]
