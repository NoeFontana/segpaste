"""Source-selection strategies for :class:`BatchCopyPaste` (ADR-0011).

Abstracts over *where the paste sources come from* so :class:`BatchCopyPaste`
can operate either on intra-batch sources (the v0.3.0 default) or on an
external instance bank without changing its forward graph.

The protocol returns a ``(source_view, placement)`` tuple where ``source_view``
is a :class:`PaddedBatchedDenseSample` row-aligned with ``target`` (matching
``B``) and ``placement.source_idx`` indexes into the source view (not the
target). For :class:`IntraBatchSource`, ``source_view is target`` and
``source_idx`` is drawn off-diagonal — bitwise identical to v0.3.0.

The forward graph in :class:`BatchCopyPaste` calls
``self.source_strategy.sample(...)`` once and feeds the result to
:class:`AffinePropagator`. No graph branching on the strategy type — both
strategies return the same shapes.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from torch import Tensor, nn
from torchvision import tv_tensors

from segpaste._internal.gpu.batched_placement import (
    BatchedPlacement,
    BatchedPlacementConfig,
    BatchedPlacementSampler,
)
from segpaste.types import PaddedBatchedDenseSample


@runtime_checkable
class SourceStrategy(Protocol):
    """Picks the source view and per-target placement for one forward step.

    Implementations may be ``nn.Module`` subclasses (so child modules and
    buffers register correctly) or plain callables — :func:`runtime_checkable`
    structural typing only requires ``sample``. The return contract is fixed:
    ``source_view`` row-aligned with ``target`` along the batch dim, and
    ``placement.source_idx`` indexing into ``source_view``.
    """

    def sample(
        self,
        target: PaddedBatchedDenseSample,
        valid_extent: Tensor | None,
        source_eligible: Tensor | None,
        generator: torch.Generator | None,
    ) -> tuple[PaddedBatchedDenseSample, BatchedPlacement]: ...


class IntraBatchSource(nn.Module):
    """v0.3.0-equivalent source: sample sources from the same batch.

    Wraps :class:`BatchedPlacementSampler` and returns ``target`` itself as
    the source view. The diagonal-masked multinomial inside the sampler
    guarantees ``source_idx[i] != i`` for ``B > 1``. Default constructor
    matches v0.3.0 defaults; pass a :class:`BatchedPlacementConfig` for
    non-default placement parameters.
    """

    def __init__(self, config: BatchedPlacementConfig | None = None) -> None:
        super().__init__()
        self.placement_sampler = BatchedPlacementSampler(config)

    def sample(
        self,
        target: PaddedBatchedDenseSample,
        valid_extent: Tensor | None,
        source_eligible: Tensor | None,
        generator: torch.Generator | None,
    ) -> tuple[PaddedBatchedDenseSample, BatchedPlacement]:
        placement = self.placement_sampler(
            target,
            generator,
            valid_extent=valid_extent,
            source_eligible=source_eligible,
        )
        return target, placement


class BankSource(nn.Module):
    """External-bank source: paste pre-staged crops from an instance bank.

    The bank tensor (``[B, K_bank, 5, h, w]`` packed as RGB+alpha+class-id)
    is set per training step via :meth:`set_bank_batch`. ``sample`` then
    multinomial-picks one crop per batch row, places it at the origin of
    a target-sized canvas (forming a synthetic source view with
    ``K_source = 1``), and draws per-target ``(scale, translate, hflip)``
    affine parameters. ``placement.source_idx = arange(B)`` so each
    target gathers from its own row of the source view.

    Pinning ``K_source = 1`` matches v0.3.0 "one paste per target"
    semantics; multi-crop-per-target is a future extension. Configurable
    placement geometry shares :class:`BatchedPlacementConfig` knobs with
    :class:`IntraBatchSource`.
    """

    def __init__(self, placement_config: BatchedPlacementConfig | None = None) -> None:
        super().__init__()
        self.placement_config = placement_config or BatchedPlacementConfig()
        self._bank_batch: Tensor | None = None

    def set_bank_batch(self, bank_batch: Tensor) -> None:
        """Stage a per-step bank tensor of shape ``[B, K_bank, 5, h, w]``."""
        if bank_batch.ndim != 5 or bank_batch.shape[2] != 5:
            shape = tuple(bank_batch.shape)
            raise ValueError(
                f"bank_batch must be [B, K_bank, 5, h, w], got {shape}"
            )
        self._bank_batch = bank_batch

    def sample(
        self,
        target: PaddedBatchedDenseSample,
        valid_extent: Tensor | None,
        source_eligible: Tensor | None,  # noqa: ARG002 — bank ignores panoptic gating
        generator: torch.Generator | None,
    ) -> tuple[PaddedBatchedDenseSample, BatchedPlacement]:
        bank = self._bank_batch
        if bank is None:
            raise RuntimeError("BankSource requires set_bank_batch(...) before forward")
        b = target.batch_size
        if bank.shape[0] != b:
            raise ValueError(f"bank_batch B={bank.shape[0]} != target.batch_size {b}")
        k_bank = bank.shape[1]
        h_crop = bank.shape[3]
        w_crop = bank.shape[4]
        canvas_h, canvas_w = target.images.shape[-2:]
        device = bank.device

        # 1. Per-target crop selection — single uniform multinomial over K_bank.
        weights = torch.ones((b, k_bank), device=device, dtype=torch.float32)
        selected = torch.multinomial(
            weights, num_samples=1, generator=generator
        ).squeeze(-1)  # [B]
        batch_arange = torch.arange(b, device=device)
        chosen = bank[batch_arange, selected]  # [B, 5, h_crop, w_crop]

        # 2. Build the synthetic [B, ...] source view at target's canvas size.
        target_dtype = target.images.dtype
        img_canvas = torch.zeros(
            (b, 3, canvas_h, canvas_w), dtype=target_dtype, device=device
        )
        img_canvas[:, :, :h_crop, :w_crop] = chosen[:, 0:3].to(target_dtype)
        mask_canvas = torch.zeros(
            (b, 1, canvas_h, canvas_w), dtype=torch.bool, device=device
        )
        mask_canvas[:, :, :h_crop, :w_crop] = chosen[:, 3:4] > 0.5
        labels = chosen[:, 4, 0, 0].to(torch.int64).unsqueeze(1)  # [B, 1]
        boxes_xyxy = torch.zeros((b, 1, 4), dtype=torch.float32, device=device)
        boxes_xyxy[:, 0, 2] = float(w_crop)
        boxes_xyxy[:, 0, 3] = float(h_crop)
        instance_ids = torch.zeros((b, 1), dtype=torch.int32, device=device)
        instance_valid_src = torch.ones((b, 1), dtype=torch.bool, device=device)
        source_view = PaddedBatchedDenseSample(
            images=tv_tensors.Image(img_canvas),
            boxes=boxes_xyxy,
            labels=labels,
            instance_valid=instance_valid_src,
            max_instances=1,
            instance_masks=mask_canvas,
            instance_ids=instance_ids,
        )

        # 3. Per-target affine placement.
        placement = self._sample_placement(
            target,
            source_view,
            valid_extent=valid_extent,
            generator=generator,
            device=device,
            crop_h=h_crop,
            crop_w=w_crop,
        )
        return source_view, placement

    def _sample_placement(
        self,
        target: PaddedBatchedDenseSample,
        source_view: PaddedBatchedDenseSample,
        *,
        valid_extent: Tensor | None,
        generator: torch.Generator | None,
        device: torch.device,
        crop_h: int,
        crop_w: int,
    ) -> BatchedPlacement:
        """Per-target ``(scale, translate, hflip)`` draw for the bank crop.

        Mirrors the contiguous-canvas branch of :class:`BatchedPlacementSampler`
        but specialized to ``K_source = 1`` and ``source_idx = arange(B)``.
        Patch-aligned paste is deferred to a follow-up — bank geometry is
        already canvas-aligned by construction at the bank's ``(h_crop,
        w_crop)``, so there is no ``pad_to_multiple`` quantization issue.
        """
        config = self.placement_config
        b = target.batch_size
        canvas_h, canvas_w = target.images.shape[-2:]

        if valid_extent is None:
            ve = source_view.images.new_tensor(
                [float(canvas_h), float(canvas_w)]
            ).expand(b, 2)
        else:
            ve = valid_extent.to(device=device, dtype=torch.float32)
        tgt_h = ve[:, 0]
        tgt_w = ve[:, 1]

        smin, smax = config.scale_range
        scale = (
            torch.rand((b,), generator=generator, device=device) * (smax - smin) + smin
        )
        hflip = (
            torch.empty((b,), device=device, dtype=torch.float32)
            .bernoulli_(config.hflip_probability, generator=generator)
            .bool()
        )

        # Crop spans the whole [0, w_crop] x [0, h_crop] region; effective
        # right/bottom edges are the crop dimensions.
        max_scaled_x2 = float(crop_w) * scale
        max_scaled_y2 = float(crop_h) * scale
        max_ty = torch.clamp(tgt_h - max_scaled_y2, min=0.0)
        max_tx = torch.clamp(tgt_w - max_scaled_x2, min=0.0)
        ty = torch.rand((b,), generator=generator, device=device) * max_ty
        tx = torch.rand((b,), generator=generator, device=device) * max_tx
        translate = torch.stack([ty, tx], dim=-1)

        fits = (max_scaled_y2 <= tgt_h) & (max_scaled_x2 <= tgt_w)
        do_paste = (
            torch.empty((b,), device=device, dtype=torch.float32)
            .bernoulli_(config.paste_prob, generator=generator)
            .bool()
        )
        paste_valid = (fits & do_paste).unsqueeze(-1)  # [B, 1]

        source_idx = torch.arange(b, device=device, dtype=torch.int64)
        src_valid_extent = (
            source_view.images.new_tensor([float(crop_h), float(crop_w)])
            .expand(b, 2)
            .contiguous()
        )

        return BatchedPlacement(
            source_idx=source_idx,
            translate=translate,
            scale=scale,
            hflip=hflip,
            paste_valid=paste_valid,
            src_valid_extent=src_valid_extent,
        )
