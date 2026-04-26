"""Pipeline that runs a `BatchCopyPasteConfig` against a sample list.

Orchestrates: stack → pad → forward → unpad → invariants → drilldown.
The `force_overlap` kwarg is an internal-only test hook (no CLI surface)
that injects same-class instance overlap into the post-aug sample so
the failure path can be exercised end-to-end without an actual bug
in the augmentation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import torch
from torch import Tensor

from segpaste._internal.invariants._report import InvariantReport
from segpaste._internal.viz.invariant_runner import run_invariants
from segpaste._internal.viz.overlay import render_drilldown
from segpaste.augmentation.batch_copy_paste import (
    BatchCopyPaste,
    BatchCopyPasteConfig,
)
from segpaste.types import (
    BatchedDenseSample,
    DenseSample,
    InstanceMask,
)


@dataclass(frozen=True, slots=True)
class SampleOutcome:
    """One sample's contribution to the gallery."""

    index: int
    before: DenseSample
    after: DenseSample
    reports: tuple[InvariantReport, ...]
    drilldown: dict[str, Tensor]

    @property
    def ok(self) -> bool:
        return all(r.ok for r in self.reports)


def run_preset(
    config: BatchCopyPasteConfig,
    samples: list[DenseSample],
    *,
    seed: int,
    device: torch.device,
    force_overlap: bool = False,
) -> list[SampleOutcome]:
    """Run *config* through *samples* on *device*; return per-sample outcomes."""
    if not samples:
        raise ValueError("samples must be non-empty")

    samples_dev = [_move_sample(s, device) for s in samples]
    max_target = max(s.boxes.shape[0] for s in samples_dev)
    if max_target == 0:
        raise ValueError("synthetic samples must carry at least one instance")
    # 2x: target's own rows + worst-case full-batch paste from one source.
    max_instances = 2 * max_target

    batched = BatchedDenseSample.from_samples(samples_dev)
    padded = batched.to_padded(max_instances=max_instances)

    module = BatchCopyPaste(config).to(device).eval()
    generator = torch.Generator(device=device).manual_seed(seed)

    with torch.inference_mode():
        after_padded = module.forward(padded, generator=generator)
    after_samples = BatchedDenseSample.from_padded(after_padded).to_samples()

    if force_overlap:
        after_samples = [_inject_same_class_overlap(s) for s in after_samples]

    outcomes: list[SampleOutcome] = []
    for i, (before, after) in enumerate(zip(samples_dev, after_samples, strict=True)):
        before_cpu = _move_sample(before, torch.device("cpu"))
        after_cpu = _move_sample(after, torch.device("cpu"))
        reports = run_invariants(before_cpu, after_cpu)
        drilldown = render_drilldown(before_cpu, after_cpu)
        outcomes.append(
            SampleOutcome(
                index=i,
                before=before_cpu,
                after=after_cpu,
                reports=tuple(reports),
                drilldown=drilldown,
            )
        )
    return outcomes


def run_preset_batched(
    config: BatchCopyPasteConfig,
    samples: Sequence[DenseSample],
    *,
    seed: int,
    batch_size: int,
    device: torch.device,
) -> list[SampleOutcome]:
    """Run *config* over *samples* in chunks of *batch_size*.

    Per-chunk seed is offset by ``chunk_start`` so adjacent chunks are
    not bit-identical; outcome indices are remapped to the flat sample
    position so artifacts land at ``samples/{i:04d}_*.png``.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    outcomes: list[SampleOutcome] = []
    for start in range(0, len(samples), batch_size):
        chunk = list(samples[start : start + batch_size])
        chunk_out = run_preset(config, chunk, seed=seed + start, device=device)
        outcomes.extend(replace(o, index=start + o.index) for o in chunk_out)
    return outcomes


def _move_sample(sample: DenseSample, device: torch.device) -> DenseSample:
    """Return *sample* with every tensor field moved to *device*."""
    if sample.image.device == device:
        return sample
    fields_dict = sample.to_dict()
    moved = {
        k: (v.to(device) if isinstance(v, Tensor) else v)
        for k, v in fields_dict.items()
    }
    return DenseSample.from_dict(moved)


def _inject_same_class_overlap(after: DenseSample) -> DenseSample:
    """Mutate an after-sample so two instances share a class and overlap."""
    if after.instance_masks is None or after.instance_masks.size(0) < 2:
        raise ValueError(
            "force_overlap requires at least two instance masks in the after sample"
        )

    masks = after.instance_masks.as_subclass(torch.Tensor).clone()
    labels = after.labels.clone()
    labels[1] = labels[0]
    masks[0, 0, 0] = True
    masks[1, 0, 0] = True

    return replace(after, instance_masks=InstanceMask(masks), labels=labels)
