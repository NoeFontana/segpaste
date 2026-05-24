"""Implementation Protocol + registry (ADR-0016 §2)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable

import torch

from benchmarks.comparison.workload import CanonicalSample, Workload

InputBatch = Any
"""Native batch shape a given implementation consumes. Opaque to the harness."""


@runtime_checkable
class Implementation(Protocol):
    """One copy-paste implementation under comparison."""

    name: str
    """Stable identifier, used as the dict key in ``comparison_v1.implementations``."""

    def supports_device(self, device: torch.device) -> bool:
        """Whether this implementation can run on ``device``.

        Returning ``False`` causes the sweep to emit a ``status: "skipped"``
        report with ``skip_reason`` set to ``"unsupported device"``.
        """
        ...

    def adapt(self, batches: Sequence[Sequence[CanonicalSample]]) -> list[InputBatch]:
        """Convert canonical batches into the impl's native batch shape.

        Called once per workload, outside the timed window.
        """
        ...

    def step(self, batch: InputBatch) -> object:
        """Run one augmentation step. Return value is ignored by the harness."""
        ...


ImplementationFactory = Callable[[], Implementation]
"""Zero-arg factory; the registry stores factories, not instances, so we
construct fresh state per workload to keep BatchCopyPaste etc. from
carrying buffers across image sizes."""


_REGISTRY: dict[str, ImplementationFactory] = {}


def register(name: str, factory: ImplementationFactory) -> None:
    """Add an implementation factory to the registry.

    Raises:
        ValueError: when ``name`` is already registered.
    """
    if name in _REGISTRY:
        raise ValueError(f"implementation {name!r} is already registered")
    _REGISTRY[name] = factory


def registry() -> dict[str, ImplementationFactory]:
    """Snapshot of the current registry, sorted by name."""
    return dict(sorted(_REGISTRY.items()))


def build(name: str) -> Implementation:
    """Look up an implementation by name and construct a fresh instance."""
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown implementation {name!r}; registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]()


__all__ = [
    "Implementation",
    "ImplementationFactory",
    "InputBatch",
    "Workload",
    "build",
    "register",
    "registry",
]
