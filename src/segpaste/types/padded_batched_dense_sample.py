"""Padded batched dense-sample container.

Sibling to :class:`BatchedDenseSample` introduced by ADR-0008. Replaces the
ragged per-sample ``list`` fields (``boxes``, ``labels``, ``instance_masks``,
``instance_ids``, ``camera_intrinsics``) with K-padded tensors gated by a
leading-B :attr:`instance_valid` mask. Every field is a leading-B tensor, so
the container is consumable from inside a ``torch.compile(fullgraph=True)``
region without Python-level iteration over the batch.

Instance-side rows where ``instance_valid`` is ``False`` must always be
zero-valued — downstream kernels gate writes on the validity mask and assume
invalid slots contribute nothing to the composite.
"""

from dataclasses import dataclass

import torch
from torchvision import tv_tensors

from segpaste.compile_util import skip_if_compiling
from segpaste.types.data_structures import PaddingMask
from segpaste.types.dense_sample import PanopticMap, SemanticMap


@dataclass(frozen=True, slots=True)
class PaddedBatchedDenseSample:
    """Fully-rectangular batched container for graph-compilable augmentation."""

    images: tv_tensors.Image  # [B, C, H, W]
    boxes: torch.Tensor  # [B, K, 4] float, xyxy; invalid rows zeroed.
    labels: torch.Tensor  # [B, K] int64; invalid rows zeroed.
    instance_valid: torch.Tensor  # [B, K] bool
    max_instances: int
    instance_masks: torch.Tensor | None = None  # [B, K, H, W] bool
    instance_ids: torch.Tensor | None = None  # [B, K] int32
    semantic_maps: SemanticMap | None = None  # [B, H, W] int64
    panoptic_maps: PanopticMap | None = None  # [B, H, W] int64
    depth: torch.Tensor | None = None  # [B, 1, H, W] float32
    depth_valid: torch.Tensor | None = None  # [B, 1, H, W] bool
    normals: torch.Tensor | None = None  # [B, 3, H, W] float32
    padding_mask: PaddingMask | None = None  # [B, 1, H, W] bool
    camera_intrinsics: torch.Tensor | None = None  # [B, 4] float32 (fx, fy, cx, cy)

    @skip_if_compiling
    def __post_init__(self) -> None:
        b = self.images.size(0)
        k = self.max_instances
        if self.images.ndim != 4:
            raise ValueError("images must have rank 4 [B, C, H, W]")
        h, w = self.images.shape[-2:]

        if self.boxes.shape != (b, k, 4):
            raise ValueError("boxes must have shape [B, K, 4]")
        if self.labels.shape != (b, k):
            raise ValueError("labels must have shape [B, K]")
        if self.instance_valid.shape != (b, k):
            raise ValueError("instance_valid must have shape [B, K]")
        if self.instance_valid.dtype != torch.bool:
            raise ValueError("instance_valid dtype must be bool")

        if (self.instance_masks is None) ^ (self.instance_ids is None):
            raise ValueError(
                "instance_masks and instance_ids must both be set or both None"
            )
        if self.instance_masks is not None and self.instance_ids is not None:
            if self.instance_masks.shape != (b, k, h, w):
                raise ValueError("instance_masks must have shape [B, K, H, W]")
            if self.instance_masks.dtype != torch.bool:
                raise ValueError("instance_masks dtype must be bool")
            if self.instance_ids.shape != (b, k):
                raise ValueError("instance_ids must have shape [B, K]")
            if self.instance_ids.dtype != torch.int32:
                raise ValueError("instance_ids dtype must be int32")

        if (self.depth is None) ^ (self.depth_valid is None):
            raise ValueError("depth and depth_valid must both be set or both None")

        stacked_shape_checks = (
            ("semantic_maps", self.semantic_maps, 3),
            ("panoptic_maps", self.panoptic_maps, 3),
            ("depth", self.depth, 4),
            ("depth_valid", self.depth_valid, 4),
            ("normals", self.normals, 4),
            ("padding_mask", self.padding_mask, 4),
        )
        for name, tensor, expected_rank in stacked_shape_checks:
            if tensor is None:
                continue
            if tensor.ndim != expected_rank:
                raise ValueError(f"{name} must have rank {expected_rank}")
            if tensor.size(0) != b:
                raise ValueError(f"{name} must have batch dim {b}")
            if tensor.shape[-2:] != (h, w):
                raise ValueError(f"{name} must share (H, W) with images")

        if self.camera_intrinsics is not None and self.camera_intrinsics.shape != (
            b,
            4,
        ):
            raise ValueError("camera_intrinsics must have shape [B, 4]")

    @property
    def batch_size(self) -> int:
        return self.images.size(0)
