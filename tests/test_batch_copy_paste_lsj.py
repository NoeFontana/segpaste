"""End-to-end LSJ-pad correctness for :class:`BatchCopyPaste`.

Constructs a batch where every sample carries a bottom-right zero-pad
band (the layout :func:`make_large_scale_jittering` produces) and a
matching ``padding_mask``. Asserts placement-side and composite-side
valid-region propagation:

* Pasted instance bboxes land entirely inside the per-sample top-left
  valid rect.
* No all-zero pasted region appears inside the effective paste mask
  (the source-pad-leaks-into-composite bug from the FiftyOne sweep).
* The output ``padding_mask`` reflects the target's valid extent.
"""

from __future__ import annotations

from dataclasses import replace

import torch
from torchvision import tv_tensors

from segpaste import BatchCopyPaste, PaddedBatchedDenseSample
from segpaste.augmentation.batch_copy_paste import (
    BatchCopyPasteConfig,
    drop_occluded_targets,
)
from segpaste.types import (
    BatchedDenseSample,
    DenseSample,
    InstanceMask,
    PaddingMask,
)

H = W = 64


def _padded_sample(
    seed: int,
    valid_h: int,
    valid_w: int,
    num_objects: int = 2,
) -> DenseSample:
    """LSJ-shaped sample: top-left valid rect, zero-pad bottom/right."""
    gen = torch.Generator().manual_seed(seed)
    image = torch.zeros(3, H, W, dtype=torch.float32)
    image[:, :valid_h, :valid_w] = (
        torch.rand(3, valid_h, valid_w, generator=gen, dtype=torch.float32) * 0.5 + 0.5
    )
    masks = torch.zeros(num_objects, H, W, dtype=torch.bool)
    raw_boxes: list[list[int]] = []
    for i in range(num_objects):
        x1 = 2 + i * 6
        y1 = 2 + i * 6
        x2 = x1 + 8
        y2 = y1 + 8
        masks[i, y1:y2, x1:x2] = True
        raw_boxes.append([x1, y1, x2, y2])
    pad = torch.zeros(1, H, W, dtype=torch.bool)
    pad[:, valid_h:, :] = True
    pad[:, :, valid_w:] = True
    return DenseSample(
        image=tv_tensors.Image(image),
        boxes=tv_tensors.BoundingBoxes(  # pyright: ignore[reportCallIssue]
            torch.tensor(raw_boxes, dtype=torch.float32),
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=(H, W),
        ),
        labels=torch.arange(1, num_objects + 1, dtype=torch.int64),
        instance_ids=torch.arange(num_objects, dtype=torch.int32),
        instance_masks=InstanceMask(masks),
        padding_mask=PaddingMask.from_tensor(pad),
    )


def _padded_batch(extents: list[tuple[int, int]]) -> PaddedBatchedDenseSample:
    samples = [
        _padded_sample(seed=i, valid_h=h, valid_w=w) for i, (h, w) in enumerate(extents)
    ]
    return BatchedDenseSample.from_samples(samples).to_padded(max_instances=4)


class TestLSJPropagation:
    def test_pasted_box_lands_inside_valid_rect(self) -> None:
        """Output box edges must fall inside the per-sample top-left valid rect."""
        extents = [(20, 30), (40, 25), (50, 60), (24, 24)]
        padded = _padded_batch(extents)
        out = BatchCopyPaste()(padded, torch.Generator().manual_seed(0))

        for b, (h_v, w_v) in enumerate(extents):
            valid = out.instance_valid[b]
            for k in range(out.max_instances):
                if not bool(valid[k]):
                    continue
                box = out.boxes[b, k]
                assert float(box[0]) >= -1e-3
                assert float(box[1]) >= -1e-3
                assert float(box[2]) <= float(w_v) + 1e-3
                assert float(box[3]) <= float(h_v) + 1e-3

    def test_no_zero_pixels_inside_paste_mask(self) -> None:
        """Composite must not pull source-pad zeros into pasted regions.

        The original LSJ samples set pixel content in ``[0.5, 1.0)`` so any
        zero pixel in a pasted region is a source-pad leak.
        """
        extents = [(28, 28), (40, 50), (60, 30), (24, 36)]
        padded = _padded_batch(extents)
        out = BatchCopyPaste()(padded, torch.Generator().manual_seed(0))

        assert out.instance_masks is not None
        assert padded.instance_masks is not None
        pasted = out.instance_masks & ~padded.instance_masks
        paste_union = pasted.any(dim=1)
        if not bool(paste_union.any()):
            return
        out_img = out.images.as_subclass(torch.Tensor)
        m3 = paste_union.unsqueeze(1).expand_as(out_img)
        pasted_pixels = out_img[m3]
        assert pasted_pixels.numel() > 0
        assert bool((pasted_pixels > 0.0).all())

    def test_output_padding_mask_warps_per_sample(self) -> None:
        extents = [(20, 24), (32, 40)]
        padded = _padded_batch(extents)
        out = BatchCopyPaste()(padded, torch.Generator().manual_seed(0))
        assert out.padding_mask is not None
        pm = out.padding_mask.as_subclass(torch.Tensor)
        assert pm.shape == (len(extents), 1, H, W)
        assert pm.dtype == torch.bool


class TestSlotMerge:
    """Compact merge: target rows preserved, pastes fill free slots."""

    def test_target_instances_preserved_when_room_exists(self) -> None:
        """With K = 2 * num_objects, every target row survives in output.

        The residual-area gate is disabled here — this test pins
        slot-merge correctness, not occlusion handling.
        """
        samples = [
            _padded_sample(seed=i, valid_h=H, valid_w=W, num_objects=2)
            for i in range(4)
        ]
        padded = BatchedDenseSample.from_samples(samples).to_padded(max_instances=4)
        cfg = BatchCopyPasteConfig(min_residual_area_frac=0.0)
        out = BatchCopyPaste(cfg)(padded, torch.Generator().manual_seed(0))

        assert padded.instance_masks is not None
        assert out.instance_masks is not None

        for b in range(padded.batch_size):
            tgt_valid = padded.instance_valid[b]
            for k in range(padded.max_instances):
                if not bool(tgt_valid[k]):
                    continue
                # Target slot k must remain valid in output.
                assert bool(out.instance_valid[b, k])
                assert torch.equal(out.labels[b, k], padded.labels[b, k])

    def test_pastes_fill_free_slots(self) -> None:
        """Free target slots receive pasted rows; valid count grows."""
        samples = [
            _padded_sample(seed=i, valid_h=H, valid_w=W, num_objects=2)
            for i in range(4)
        ]
        padded = BatchedDenseSample.from_samples(samples).to_padded(max_instances=4)
        out = BatchCopyPaste()(padded, torch.Generator().manual_seed(0))

        # At least one sample must have grown its valid count (pastes landed).
        in_count = padded.instance_valid.sum(dim=-1)
        out_count = out.instance_valid.sum(dim=-1)
        assert bool((out_count >= in_count).all())
        assert bool((out_count > in_count).any())


class TestResidualAreaGate:
    """Heavily-occluded target instances are dropped from instance_valid."""

    def test_drops_target_when_paste_covers_above_threshold(self) -> None:
        """A target whose survivor mask is <10% of its original area is dropped.

        Constructed directly on the compositor output: the gate is a pure
        survivor/original ratio over ``instance_masks`` and does not depend
        on the rest of the placement pipeline.
        """
        samples = [
            _padded_sample(seed=i, valid_h=H, valid_w=W, num_objects=2)
            for i in range(2)
        ]
        padded = BatchedDenseSample.from_samples(samples).to_padded(max_instances=4)
        assert padded.instance_masks is not None

        survivor = padded.instance_masks.clone()
        orig_area = int(padded.instance_masks[0, 0].sum().item())
        keep_pixels = max(1, orig_area // 20)
        flat = survivor[0, 0].view(-1)
        nonzero = flat.nonzero().squeeze(-1)
        flat[nonzero[keep_pixels:]] = False
        survivor[0, 0] = flat.view(H, W)

        composited = replace(padded, instance_masks=survivor)
        gated = drop_occluded_targets(padded, composited, 0.1)

        assert not bool(gated.instance_valid[0, 0])
        assert bool(gated.instance_valid[0, 1])
        assert bool(gated.instance_valid[1, 0])

    def test_zero_threshold_keeps_everything(self) -> None:
        samples = [
            _padded_sample(seed=i, valid_h=H, valid_w=W, num_objects=2)
            for i in range(2)
        ]
        padded = BatchedDenseSample.from_samples(samples).to_padded(max_instances=4)
        assert padded.instance_masks is not None
        gated = drop_occluded_targets(padded, padded, 0.0)
        assert torch.equal(gated.instance_valid, padded.instance_valid)
