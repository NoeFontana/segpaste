"""Tests for :class:`segpaste._internal.composite.DenseComposite`."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from torchvision import tv_tensors

from segpaste._internal.composite import CompositeConfig, DenseComposite
from segpaste.types import (
    DenseSample,
    InstanceMask,
    Modality,
    PaddingMask,
    PanopticMap,
    SemanticMap,
)
from tests.strategies import dense_sample_strategy


def _mk_instance_sample(
    h: int, w: int, boxes: list[tuple[int, int, int, int]], seed: int = 0
) -> DenseSample:
    g = torch.Generator().manual_seed(seed)
    image = torch.rand(3, h, w, generator=g, dtype=torch.float32)
    n = len(boxes)
    masks = torch.zeros(n, h, w, dtype=torch.bool)
    box_t = torch.zeros(n, 4, dtype=torch.float32)
    for i, (y1, x1, y2, x2) in enumerate(boxes):
        masks[i, y1:y2, x1:x2] = True
        box_t[i] = torch.tensor([x1, y1, x2, y2])
    return DenseSample(
        image=tv_tensors.Image(image),
        boxes=tv_tensors.BoundingBoxes(  # pyright: ignore[reportCallIssue]
            box_t, format=tv_tensors.BoundingBoxFormat.XYXY, canvas_size=(h, w)
        ),
        labels=torch.arange(1, n + 1, dtype=torch.int64),
        instance_ids=torch.arange(n, dtype=torch.int32),
        instance_masks=InstanceMask(masks),
    )


def _cfg(**kwargs: object) -> CompositeConfig:
    return CompositeConfig(min_composited_area=0, **kwargs)  # pyright: ignore[reportArgumentType]


class TestImageComposite:
    def test_alpha_matches_manual_blend(self) -> None:
        """Parity math: ``tgt*(1-m) + src*m`` for bool-cast mask."""
        tgt = _mk_instance_sample(16, 16, [(2, 2, 6, 6)], seed=0)
        src = _mk_instance_sample(16, 16, [(8, 8, 12, 12)], seed=1)
        paste = torch.zeros(16, 16, dtype=torch.bool)
        paste[8:12, 8:12] = True

        out = DenseComposite(_cfg())(tgt, src, paste)

        m = paste.to(torch.float32).unsqueeze(0)
        expected = (
            tgt.image.as_subclass(torch.Tensor) * (1.0 - m)
            + src.image.as_subclass(torch.Tensor) * m
        )
        assert torch.equal(out.image.as_subclass(torch.Tensor), expected)

    def test_empty_paste_mask_preserves_target_image(self) -> None:
        tgt = _mk_instance_sample(16, 16, [(2, 2, 6, 6)])
        src = _mk_instance_sample(16, 16, [(8, 8, 12, 12)], seed=1)
        paste = torch.zeros(16, 16, dtype=torch.bool)

        out = DenseComposite(_cfg())(tgt, src, paste)

        assert torch.equal(
            out.image.as_subclass(torch.Tensor),
            tgt.image.as_subclass(torch.Tensor),
        )


class TestInstanceComposite:
    def test_survivor_masks_subtract_paste_union(self) -> None:
        """Target masks are subtracted by the paste mask (ADR-0001 §(ii))."""
        tgt = _mk_instance_sample(16, 16, [(2, 2, 10, 10)])
        src = _mk_instance_sample(16, 16, [(8, 8, 12, 12)], seed=1)
        paste = torch.zeros(16, 16, dtype=torch.bool)
        paste[8:12, 8:12] = True

        out = DenseComposite(_cfg())(tgt, src, paste)

        assert out.instance_masks is not None
        # Row 0 is the survivor (still has area outside the paste).
        assert not out.instance_masks[0, 8:12, 8:12].any()
        # Row 1 is the paste.
        assert out.instance_masks[1, 8:12, 8:12].all()

    def test_fresh_instance_ids_allocated(self) -> None:
        tgt = _mk_instance_sample(16, 16, [(2, 2, 6, 6), (8, 0, 12, 4)])
        src = _mk_instance_sample(16, 16, [(8, 8, 12, 12)], seed=1)
        paste = torch.zeros(16, 16, dtype=torch.bool)
        paste[8:12, 8:12] = True

        out = DenseComposite(_cfg())(tgt, src, paste)

        assert out.instance_ids is not None
        assert out.instance_ids.tolist() == [0, 1, 2]

    def test_boxes_consistent_with_masks(self) -> None:
        tgt = _mk_instance_sample(16, 16, [(2, 2, 6, 6)])
        src = _mk_instance_sample(16, 16, [(8, 8, 12, 12)], seed=1)
        paste = torch.zeros(16, 16, dtype=torch.bool)
        paste[8:12, 8:12] = True

        out = DenseComposite(_cfg())(tgt, src, paste)

        from torchvision.ops import masks_to_boxes

        assert out.instance_masks is not None
        expected = masks_to_boxes(out.instance_masks.as_subclass(torch.Tensor))
        assert torch.equal(out.boxes.as_subclass(torch.Tensor), expected)

    def test_small_area_dropped(self) -> None:
        """Survivor with ``updated_area < threshold_ratio * original_area`` drops."""
        # Target mask is 4x4=16 px; paste covers 15 of them → 93.75% occluded.
        tgt = _mk_instance_sample(16, 16, [(0, 0, 4, 4)])
        src = _mk_instance_sample(16, 16, [(8, 8, 12, 12)], seed=1)
        paste = torch.zeros(16, 16, dtype=torch.bool)
        paste[0:4, 0:3] = True  # 12 px → 75% occluded, survives default
        paste[0:3, 3:4] = True  # 15 px → 93.75% occluded, survives default

        cfg = CompositeConfig(min_composited_area=0, occluded_area_threshold=0.5)
        out = DenseComposite(cfg)(tgt, src, paste)

        # Threshold 0.5 → 93.75%-occluded survivor drops; source paste row survives.
        assert out.instance_masks is not None
        assert out.instance_masks.shape[0] == 1
        assert out.instance_ids is not None
        # Paste row gets max_prev+1 = 1 (max_prev is derived pre-filter).
        assert out.instance_ids.tolist() == [1]


class TestEffectiveMask:
    def test_no_depth_eff_equals_paste(self) -> None:
        tgt = _mk_instance_sample(16, 16, [(2, 2, 6, 6)])
        src = _mk_instance_sample(16, 16, [(8, 8, 12, 12)], seed=1)
        paste = torch.zeros(16, 16, dtype=torch.bool)
        paste[8:12, 8:12] = True

        composite = DenseComposite(_cfg())
        m_eff = composite._effective_mask(tgt, src, paste)  # pyright: ignore[reportPrivateUsage]

        assert torch.equal(m_eff, paste)

    def test_depth_z_buffer_drops_farther_source(self) -> None:
        """Source pixels farther than target are masked out."""
        h, w = 8, 8
        tgt_depth = torch.full((1, h, w), 1.0)  # target at depth 1
        src_depth = torch.full((1, h, w), 2.0)  # source farther
        tgt = DenseSample(
            image=tv_tensors.Image(torch.zeros(3, h, w)),
            boxes=tv_tensors.BoundingBoxes(  # pyright: ignore[reportCallIssue]
                torch.zeros(0, 4),
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=(h, w),
            ),
            labels=torch.zeros(0, dtype=torch.int64),
            depth=tgt_depth,
            depth_valid=torch.ones(1, h, w, dtype=torch.bool),
        )
        src = DenseSample(
            image=tv_tensors.Image(torch.ones(3, h, w)),
            boxes=tv_tensors.BoundingBoxes(  # pyright: ignore[reportCallIssue]
                torch.zeros(0, 4),
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=(h, w),
            ),
            labels=torch.zeros(0, dtype=torch.int64),
            depth=src_depth,
            depth_valid=torch.ones(1, h, w, dtype=torch.bool),
        )

        paste = torch.ones(h, w, dtype=torch.bool)
        composite = DenseComposite(_cfg())
        m_eff = composite._effective_mask(tgt, src, paste)  # pyright: ignore[reportPrivateUsage]

        assert not m_eff.any(), "source farther than target → z-buffer drops it"

    def test_depth_invalid_target_passes_source(self) -> None:
        h, w = 8, 8
        tgt_depth = torch.full((1, h, w), 0.5)
        src_depth = torch.full((1, h, w), 2.0)  # farther, but target invalid
        tgt_valid = torch.zeros(1, h, w, dtype=torch.bool)
        tgt = DenseSample(
            image=tv_tensors.Image(torch.zeros(3, h, w)),
            boxes=tv_tensors.BoundingBoxes(  # pyright: ignore[reportCallIssue]
                torch.zeros(0, 4),
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=(h, w),
            ),
            labels=torch.zeros(0, dtype=torch.int64),
            depth=tgt_depth,
            depth_valid=tgt_valid,
        )
        src = DenseSample(
            image=tv_tensors.Image(torch.ones(3, h, w)),
            boxes=tv_tensors.BoundingBoxes(  # pyright: ignore[reportCallIssue]
                torch.zeros(0, 4),
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=(h, w),
            ),
            labels=torch.zeros(0, dtype=torch.int64),
            depth=src_depth,
            depth_valid=torch.ones(1, h, w, dtype=torch.bool),
        )

        paste = torch.ones(h, w, dtype=torch.bool)
        composite = DenseComposite(_cfg())
        m_eff = composite._effective_mask(tgt, src, paste)  # pyright: ignore[reportPrivateUsage]

        assert m_eff.all(), "invalid target depth → paste wins regardless"

    def test_source_padding_mask_excludes_pad_pixels(self) -> None:
        """Source-pad pixels (zeros from LSJ) must not leak into the composite."""
        h = w = 16
        tgt = _mk_instance_sample(h, w, [(2, 2, 6, 6)], seed=0)
        src = _mk_instance_sample(h, w, [(0, 0, 12, 12)], seed=1)
        src_pad = torch.zeros(1, h, w, dtype=torch.bool)
        src_pad[0, 8:, 8:] = True
        src_with_pad = replace(src, padding_mask=PaddingMask(src_pad))
        paste = torch.zeros(h, w, dtype=torch.bool)
        paste[4:14, 4:14] = True

        composite = DenseComposite(_cfg())
        m_eff = composite._effective_mask(tgt, src_with_pad, paste)  # pyright: ignore[reportPrivateUsage]

        assert not bool((m_eff & src_pad.squeeze(0)).any())
        # Non-pad paste pixels still active.
        assert bool((m_eff & paste & ~src_pad.squeeze(0)).any())

    def test_source_padding_mask_keeps_target_inside_pad_overlap(self) -> None:
        """End-to-end: composed image equals target where source-pad overlaps paste."""
        h = w = 16
        tgt = _mk_instance_sample(h, w, [(2, 2, 6, 6)], seed=0)
        src = _mk_instance_sample(h, w, [(0, 0, 12, 12)], seed=1)
        src_pad = torch.zeros(1, h, w, dtype=torch.bool)
        src_pad[0, 8:, 8:] = True
        src_with_pad = replace(src, padding_mask=PaddingMask(src_pad))
        paste = torch.zeros(h, w, dtype=torch.bool)
        paste[4:14, 4:14] = True

        out = DenseComposite(_cfg())(tgt, src_with_pad, paste)

        out_t = out.image.as_subclass(torch.Tensor)
        tgt_t = tgt.image.as_subclass(torch.Tensor)
        src_t = src.image.as_subclass(torch.Tensor)
        pad_overlap = paste & src_pad.squeeze(0)
        nonpad_paste = paste & ~src_pad.squeeze(0)
        assert torch.equal(
            out_t[:, pad_overlap].flatten(), tgt_t[:, pad_overlap].flatten()
        )
        assert torch.equal(
            out_t[:, nonpad_paste].flatten(), src_t[:, nonpad_paste].flatten()
        )


class TestSemanticComposite:
    def test_where_source_into_target(self) -> None:
        h, w = 8, 8
        tgt_sem = torch.zeros(h, w, dtype=torch.int64)
        src_sem = torch.full((h, w), 7, dtype=torch.int64)

        tgt = DenseSample(
            image=tv_tensors.Image(torch.zeros(3, h, w)),
            boxes=tv_tensors.BoundingBoxes(  # pyright: ignore[reportCallIssue]
                torch.zeros(0, 4),
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=(h, w),
            ),
            labels=torch.zeros(0, dtype=torch.int64),
            semantic_map=SemanticMap(tgt_sem),
        )
        src = DenseSample(
            image=tv_tensors.Image(torch.zeros(3, h, w)),
            boxes=tv_tensors.BoundingBoxes(  # pyright: ignore[reportCallIssue]
                torch.zeros(0, 4),
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=(h, w),
            ),
            labels=torch.zeros(0, dtype=torch.int64),
            semantic_map=SemanticMap(src_sem),
        )

        paste = torch.zeros(h, w, dtype=torch.bool)
        paste[2:5, 2:5] = True
        out = DenseComposite(_cfg())(tgt, src, paste)

        assert out.semantic_map is not None
        sem_out = out.semantic_map.as_subclass(torch.Tensor)
        assert (sem_out[2:5, 2:5] == 7).all()
        # Complement stays at 0.
        comp = torch.ones(h, w, dtype=torch.bool)
        comp[2:5, 2:5] = False
        assert (sem_out[comp] == 0).all()

    def test_semantic_mismatch_raises(self) -> None:
        """One-sided semantic is a configuration error."""
        h, w = 8, 8
        tgt = DenseSample(
            image=tv_tensors.Image(torch.zeros(3, h, w)),
            boxes=tv_tensors.BoundingBoxes(  # pyright: ignore[reportCallIssue]
                torch.zeros(0, 4),
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=(h, w),
            ),
            labels=torch.zeros(0, dtype=torch.int64),
            semantic_map=SemanticMap(torch.zeros(h, w, dtype=torch.int64)),
        )
        src = DenseSample(
            image=tv_tensors.Image(torch.zeros(3, h, w)),
            boxes=tv_tensors.BoundingBoxes(  # pyright: ignore[reportCallIssue]
                torch.zeros(0, 4),
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=(h, w),
            ),
            labels=torch.zeros(0, dtype=torch.int64),
        )
        paste = torch.ones(h, w, dtype=torch.bool)
        with pytest.raises(ValueError):
            DenseComposite(_cfg())(tgt, src, paste)


class TestPanopticComposite:
    def test_where_source_into_target(self) -> None:
        h, w = 8, 8
        tgt_pan = torch.zeros(h, w, dtype=torch.int64)
        src_pan = torch.full((h, w), 5, dtype=torch.int64)

        tgt = DenseSample(
            image=tv_tensors.Image(torch.zeros(3, h, w)),
            boxes=tv_tensors.BoundingBoxes(  # pyright: ignore[reportCallIssue]
                torch.zeros(0, 4),
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=(h, w),
            ),
            labels=torch.zeros(0, dtype=torch.int64),
            panoptic_map=PanopticMap(tgt_pan),
        )
        src = DenseSample(
            image=tv_tensors.Image(torch.zeros(3, h, w)),
            boxes=tv_tensors.BoundingBoxes(  # pyright: ignore[reportCallIssue]
                torch.zeros(0, 4),
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=(h, w),
            ),
            labels=torch.zeros(0, dtype=torch.int64),
            panoptic_map=PanopticMap(src_pan),
        )

        paste = torch.zeros(h, w, dtype=torch.bool)
        paste[3:6, 3:6] = True
        out = DenseComposite(_cfg())(tgt, src, paste)

        assert out.panoptic_map is not None
        pan_out = out.panoptic_map.as_subclass(torch.Tensor)
        assert (pan_out[3:6, 3:6] == 5).all()
        comp = torch.ones(h, w, dtype=torch.bool)
        comp[3:6, 3:6] = False
        assert (pan_out[comp] == 0).all()

    def test_panoptic_mismatch_raises(self) -> None:
        """One-sided panoptic is a configuration error."""
        h, w = 8, 8
        tgt = DenseSample(
            image=tv_tensors.Image(torch.zeros(3, h, w)),
            boxes=tv_tensors.BoundingBoxes(  # pyright: ignore[reportCallIssue]
                torch.zeros(0, 4),
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=(h, w),
            ),
            labels=torch.zeros(0, dtype=torch.int64),
            panoptic_map=PanopticMap(torch.zeros(h, w, dtype=torch.int64)),
        )
        src = DenseSample(
            image=tv_tensors.Image(torch.zeros(3, h, w)),
            boxes=tv_tensors.BoundingBoxes(  # pyright: ignore[reportCallIssue]
                torch.zeros(0, 4),
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=(h, w),
            ),
            labels=torch.zeros(0, dtype=torch.int64),
        )
        paste = torch.ones(h, w, dtype=torch.bool)
        with pytest.raises(ValueError):
            DenseComposite(_cfg())(tgt, src, paste)

    def test_no_panoptic_returns_none(self) -> None:
        """Neither-side panoptic leaves the output's panoptic_map as None."""
        tgt = _mk_instance_sample(16, 16, [(2, 2, 6, 6)])
        src = _mk_instance_sample(16, 16, [(8, 8, 12, 12)], seed=1)
        paste = torch.zeros(16, 16, dtype=torch.bool)
        paste[8:12, 8:12] = True

        out = DenseComposite(_cfg())(tgt, src, paste)

        assert out.panoptic_map is None


class TestValidation:
    def test_wrong_dtype_paste_mask(self) -> None:
        tgt = _mk_instance_sample(16, 16, [(2, 2, 6, 6)])
        src = _mk_instance_sample(16, 16, [(8, 8, 12, 12)], seed=1)
        paste = torch.zeros(16, 16, dtype=torch.int32)
        with pytest.raises(ValueError, match="bool"):
            DenseComposite(_cfg())(tgt, src, paste)  # pyright: ignore[reportArgumentType]

    def test_wrong_shape_paste_mask(self) -> None:
        tgt = _mk_instance_sample(16, 16, [(2, 2, 6, 6)])
        src = _mk_instance_sample(16, 16, [(8, 8, 12, 12)], seed=1)
        paste = torch.zeros(16, 15, dtype=torch.bool)
        with pytest.raises(ValueError, match="H, W"):
            DenseComposite(_cfg())(tgt, src, paste)


@settings(deadline=None, max_examples=30)
@given(data=st.data())
def test_output_image_shape_preserved(data: st.DataObject) -> None:
    tgt = data.draw(dense_sample_strategy({Modality.INSTANCE}))
    src = data.draw(dense_sample_strategy({Modality.INSTANCE}))
    h, w = tgt.image.shape[-2:]
    if src.image.shape[-2:] != (h, w):
        pytest.skip("strategies can draw different sizes")
    paste = torch.zeros(h, w, dtype=torch.bool)
    # Mark a small patch as paste — exact location doesn't matter for shape.
    paste[: max(1, h // 4), : max(1, w // 4)] = True

    out = DenseComposite(_cfg())(tgt, src, paste)

    assert out.image.shape == tgt.image.shape
    assert out.instance_masks is not None
    assert out.instance_masks.shape[-2:] == (h, w)
