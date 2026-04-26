"""Panoptic-mode :class:`BatchCopyPaste` tests (ADR-0006)."""

from __future__ import annotations

from dataclasses import replace

import torch
from torchvision import tv_tensors

from segpaste._internal.gpu.batched_placement import BatchedPlacementConfig
from segpaste._internal.invariants.panoptic import (
    assert_panoptic_pixel_bijection,
    assert_panoptic_thing_stuff_consistent,
    check_panoptic_stuff_area_threshold,
)
from segpaste.augmentation.batch_copy_paste import (
    BatchCopyPaste,
    BatchCopyPasteConfig,
    PanopticPasteConfig,
)
from segpaste.types import (
    BatchedDenseSample,
    DenseSample,
    InstanceMask,
    PanopticMap,
    PanopticSchemaSpec,
    SemanticMap,
)
from tests.fixtures.loader import load_fixture
from tests.shared import make_disjoint_panoptic_sample, make_thing_stuff_schema

_STUFF_CLASS = 0


def _schema() -> PanopticSchemaSpec:
    return make_thing_stuff_schema(max_instances_per_image=4)


def _panoptic_batch(b: int = 4) -> BatchedDenseSample:
    base = load_fixture("panoptic_stuff_and_things")
    samples = [base for _ in range(b)]
    return BatchedDenseSample.from_samples(samples)


def _panoptic_config(
    *,
    paste_prob: float = 1.0,
    tau_stuff_frac: float = 0.1,
    scale: tuple[float, float] = (1.0, 1.0),
    hflip: float = 0.0,
) -> BatchCopyPasteConfig:
    return BatchCopyPasteConfig(
        placement=BatchedPlacementConfig(
            scale_range=scale,
            hflip_probability=hflip,
            paste_prob=paste_prob,
        ),
        min_residual_area_frac=0.0,
        panoptic=PanopticPasteConfig(taxonomy=_schema(), tau_stuff_frac=tau_stuff_frac),
    )


class TestThingOnlyPasteSource:
    def test_stuff_labeled_rows_are_never_pasted(self) -> None:
        # Anomalous: slot 1 carries a stuff-class label; the thing-only filter
        # must mask it out so paste_valid for that source slot stays False.
        base = make_disjoint_panoptic_sample()
        sample = replace(
            base, labels=torch.tensor([1, _STUFF_CLASS], dtype=torch.int64)
        )
        padded = BatchedDenseSample.from_samples([sample, sample]).to_padded(
            max_instances=2
        )
        mod = BatchCopyPaste(_panoptic_config())
        eligible = mod._source_eligible(padded)  # pyright: ignore[reportPrivateUsage]
        assert eligible is not None
        assert bool(eligible[0, 0].item()) is True
        assert bool(eligible[0, 1].item()) is False


class TestTauStuffRevert:
    def test_stuff_collapse_reverts_pixels(self) -> None:
        # Source covers the whole image; without revert the stuff (class 0)
        # would drop to 0 pixels. tau_stuff_frac=0.5 forces revert.
        h, w = 32, 32
        masks_full = torch.zeros((1, h, w), dtype=torch.bool)
        masks_full[0] = True
        masks_empty = torch.zeros((1, h, w), dtype=torch.bool)
        boxes_full = torch.tensor([[0.0, 0.0, float(w), float(h)]], dtype=torch.float32)
        boxes_empty = torch.zeros((1, 4), dtype=torch.float32)

        sample_a = DenseSample(
            image=tv_tensors.Image(torch.zeros(3, h, w)),
            boxes=tv_tensors.BoundingBoxes(  # pyright: ignore[reportCallIssue]
                boxes_empty,
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=(h, w),
            ),
            labels=torch.tensor([0], dtype=torch.int64),
            instance_ids=torch.tensor([0], dtype=torch.int32),
            instance_masks=InstanceMask(masks_empty),
            semantic_map=SemanticMap(torch.zeros((h, w), dtype=torch.int64)),
            panoptic_map=PanopticMap(torch.zeros((h, w), dtype=torch.int64)),
        )
        sem_b = torch.full((h, w), 1, dtype=torch.int64)
        pan_b = torch.ones((h, w), dtype=torch.int64)
        sample_b = DenseSample(
            image=tv_tensors.Image(torch.ones(3, h, w)),
            boxes=tv_tensors.BoundingBoxes(  # pyright: ignore[reportCallIssue]
                boxes_full,
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=(h, w),
            ),
            labels=torch.tensor([1], dtype=torch.int64),
            instance_ids=torch.tensor([1], dtype=torch.int32),
            instance_masks=InstanceMask(masks_full),
            semantic_map=SemanticMap(sem_b),
            panoptic_map=PanopticMap(pan_b),
        )
        padded = BatchedDenseSample.from_samples([sample_a, sample_b]).to_padded(
            max_instances=1
        )
        mod = BatchCopyPaste(_panoptic_config(tau_stuff_frac=0.5))
        out = mod(padded, torch.Generator().manual_seed(0))

        # Sample 0 sourced from Sample 1 (whole-image thing) — revert restores stuff.
        post_sem = out.semantic_maps.as_subclass(torch.Tensor)
        sample0_stuff_pixels = int((post_sem[0] == 0).sum().item())
        assert sample0_stuff_pixels >= int(0.5 * h * w)


class TestPanopticInvariants:
    def test_invariants_pass_after_paste(self) -> None:
        padded = _panoptic_batch(b=4).to_padded(max_instances=4)
        mod = BatchCopyPaste(_panoptic_config(scale=(0.5, 0.8)))
        out = mod(padded, torch.Generator().manual_seed(0))
        samples = BatchedDenseSample.from_padded(out).to_samples()
        before = load_fixture("panoptic_stuff_and_things")
        before_pix_count = before.image.shape[-1] * before.image.shape[-2]
        tau_stuff_pixels = int(0.1 * before_pix_count)
        schema = _schema()
        for after_sample in samples:
            assert_panoptic_thing_stuff_consistent(after_sample, schema)
            assert_panoptic_pixel_bijection(after_sample)
            stuff_report = check_panoptic_stuff_area_threshold(
                before, after_sample, schema, tau_stuff_pixels
            )
            assert stuff_report.ok, stuff_report.message
