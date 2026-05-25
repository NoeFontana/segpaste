"""ADR-0018 §A1 — IdentityPropagator gathers without grid_sample.

Validates that ``skip_affine=True`` selects the gather-only propagator,
produces correct shapes, and matches the gather-only fast path's output
when ``AffinePropagator`` is run with a hand-constructed identity
placement.
"""

from __future__ import annotations

import torch
from torchvision import tv_tensors

from segpaste import BatchCopyPaste, BatchedDenseSample
from segpaste._internal.gpu.affine_propagate import (
    AffinePropagator,
    IdentityPropagator,
)
from segpaste._internal.gpu.batched_placement import BatchedPlacement
from segpaste.augmentation.batch_copy_paste import BatchCopyPasteConfig
from segpaste.types import DenseSample, InstanceMask

H = W = 64


def _sample(seed: int, num_objects: int = 2) -> DenseSample:
    gen = torch.Generator().manual_seed(seed)
    image = tv_tensors.Image(torch.rand(3, H, W, generator=gen))
    masks = torch.zeros(num_objects, H, W, dtype=torch.bool)
    raw_boxes: list[list[int]] = []
    for i in range(num_objects):
        x1 = 4 + i * 8
        y1 = 4 + i * 8
        x2, y2 = x1 + 8, y1 + 8
        masks[i, y1:y2, x1:x2] = True
        raw_boxes.append([x1, y1, x2, y2])
    return DenseSample(
        image=image,
        boxes=tv_tensors.BoundingBoxes(  # pyright: ignore[reportCallIssue]
            torch.tensor(raw_boxes, dtype=torch.float32),
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=(H, W),
        ),
        labels=torch.arange(1, num_objects + 1, dtype=torch.int64),
        instance_ids=torch.arange(num_objects, dtype=torch.int32),
        instance_masks=InstanceMask(masks),
    )


def _identity_placement(b: int, k: int, h: int, w: int) -> BatchedPlacement:
    return BatchedPlacement(
        source_idx=torch.arange(b - 1, -1, -1, dtype=torch.int64),
        translate=torch.zeros((b, 2), dtype=torch.float32),
        scale=torch.ones((b,), dtype=torch.float32),
        hflip=torch.zeros((b,), dtype=torch.bool),
        paste_valid=torch.ones((b, k), dtype=torch.bool),
        src_valid_extent=torch.tensor([[float(h), float(w)]] * b, dtype=torch.float32),
    )


def test_skip_affine_binds_identity_propagator() -> None:
    config = BatchCopyPasteConfig(skip_affine=True)
    module = BatchCopyPaste(config)
    assert isinstance(module.propagator, IdentityPropagator)


def test_default_binds_affine_propagator() -> None:
    module = BatchCopyPaste()
    assert isinstance(module.propagator, AffinePropagator)


def test_identity_propagator_matches_affine_at_identity_placement() -> None:
    """IdentityPropagator output == AffinePropagator output at identity placement.

    The AffinePropagator's ``grid_sample`` reduces to the gather path
    when scale=1, hflip=0, translate=0; IdentityPropagator skips the
    grid build and sample but should produce the same tensors.
    """
    samples = [_sample(seed=i) for i in range(4)]
    padded = BatchedDenseSample.from_samples(samples).to_padded(max_instances=4)
    placement = _identity_placement(padded.batch_size, padded.max_instances, H, W)

    affine = AffinePropagator()
    identity = IdentityPropagator()

    aff_out, _ = affine(padded, padded, placement)
    id_out, _ = identity(padded, padded, placement)

    assert torch.equal(
        aff_out.images.as_subclass(torch.Tensor),
        id_out.images.as_subclass(torch.Tensor),
    )
    assert torch.equal(aff_out.boxes, id_out.boxes)
    assert torch.equal(aff_out.labels, id_out.labels)
    assert aff_out.instance_masks is not None
    assert id_out.instance_masks is not None
    assert torch.equal(aff_out.instance_masks, id_out.instance_masks)
    assert aff_out.instance_ids is not None
    assert id_out.instance_ids is not None
    assert torch.equal(aff_out.instance_ids, id_out.instance_ids)
