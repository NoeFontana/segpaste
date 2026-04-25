"""Batched dense-sample container.

Introduced by ADR-0004. The canonical batch shape consumed by downstream
training code and by :meth:`to_padded` / :class:`BatchCopyPaste`. Stacked stacks (image,
semantic_map, panoptic_map, depth, depth_valid, normals, padding_mask) are fast
to consume on GPU; per-sample ragged structures (boxes, labels, instance_masks,
instance_ids, camera_intrinsics) preserve the per-sample object count variance
that instance segmentation requires.
"""

from dataclasses import dataclass
from typing import Any, cast

import torch
from torchvision import tv_tensors

from segpaste.compile_util import skip_if_compiling
from segpaste.types.data_structures import PaddingMask
from segpaste.types.dense_sample import (
    CameraIntrinsics,
    DenseSample,
    InstanceMask,
    Modality,
    PanopticMap,
    SemanticMap,
)
from segpaste.types.padded_batched_dense_sample import PaddedBatchedDenseSample


@dataclass(frozen=True, slots=True)
class BatchedDenseSample:
    """Canonical batched container for dense-label Copy-Paste.

    Stacked fields share ``(H, W)`` across the batch — LSJ preprocessing is
    assumed to have homogenized sample shapes. Ragged fields keep one entry
    per sample. ``B == 0`` is valid (empty batches produce zero-length lists
    and zero-batch-dim stacked tensors).
    """

    images: tv_tensors.Image
    boxes: list[tv_tensors.BoundingBoxes]
    labels: list[torch.Tensor]
    instance_masks: list[InstanceMask] | None = None
    instance_ids: list[torch.Tensor] | None = None
    semantic_maps: SemanticMap | None = None
    panoptic_maps: PanopticMap | None = None
    depth: torch.Tensor | None = None
    depth_valid: torch.Tensor | None = None
    normals: torch.Tensor | None = None
    padding_mask: PaddingMask | None = None
    camera_intrinsics: list[CameraIntrinsics] | None = None

    @skip_if_compiling
    def __post_init__(self) -> None:
        b = self.images.size(0)
        if len(self.boxes) != b or len(self.labels) != b:
            raise ValueError("boxes and labels must be length B")

        if (self.instance_masks is None) ^ (self.instance_ids is None):
            raise ValueError(
                "instance_masks and instance_ids must both be set or both None"
            )
        if (
            self.instance_masks is not None
            and self.instance_ids is not None
            and (len(self.instance_masks) != b or len(self.instance_ids) != b)
        ):
            raise ValueError("instance_masks and instance_ids must be length B")

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

        if self.camera_intrinsics is not None and len(self.camera_intrinsics) != b:
            raise ValueError("camera_intrinsics must be length B")

    @property
    def batch_size(self) -> int:
        return self.images.size(0)

    @staticmethod
    def from_samples(samples: list[DenseSample]) -> "BatchedDenseSample":
        """Stack a list of :class:`DenseSample` into a :class:`BatchedDenseSample`.

        All samples must share the same active modality set and the same
        ``(H, W)``. ``B == 0`` yields an empty-but-valid batch.
        """
        if not samples:
            return _empty_batch()

        active = samples[0].active_modalities()
        h, w = samples[0].image.shape[-2:]
        for s in samples[1:]:
            if s.active_modalities() != active:
                raise ValueError("all samples must share the same active modality set")
            if s.image.shape[-2:] != (h, w):
                raise ValueError("all samples must share (H, W)")

        images = tv_tensors.Image(
            torch.stack([s.image.as_subclass(torch.Tensor) for s in samples])
        )
        boxes = [s.boxes for s in samples]
        labels = [s.labels for s in samples]

        instance_masks: list[InstanceMask] | None = None
        instance_ids: list[torch.Tensor] | None = None
        if Modality.INSTANCE in active:
            # Co-optionality on DenseSample makes None structurally impossible here.
            instance_masks = cast(
                list[InstanceMask], [s.instance_masks for s in samples]
            )
            instance_ids = cast(list[torch.Tensor], [s.instance_ids for s in samples])

        semantic_maps = _stack_optional(samples, "semantic_map", wrapper=SemanticMap)
        panoptic_maps = _stack_optional(samples, "panoptic_map", wrapper=PanopticMap)
        depth = _stack_optional(samples, "depth")
        depth_valid = _stack_optional(samples, "depth_valid")
        normals = _stack_optional(samples, "normals")
        padding_mask = _stack_optional(samples, "padding_mask", wrapper=PaddingMask)

        intrinsics = [s.camera_intrinsics for s in samples]
        camera_intrinsics = (
            cast(list[CameraIntrinsics], intrinsics)
            if all(i is not None for i in intrinsics)
            else None
        )

        return BatchedDenseSample(
            images=images,
            boxes=boxes,
            labels=labels,
            instance_masks=instance_masks,
            instance_ids=instance_ids,
            semantic_maps=semantic_maps,
            panoptic_maps=panoptic_maps,
            depth=depth,
            depth_valid=depth_valid,
            normals=normals,
            padding_mask=padding_mask,
            camera_intrinsics=camera_intrinsics,
        )

    def to_padded(self, max_instances: int) -> PaddedBatchedDenseSample:
        """Pack ragged per-sample instance fields into K-padded tensors.

        Valid rows are written at slots ``[0, n_i)`` for each sample ``i`` and
        marked ``True`` in ``instance_valid``. Padded rows are zero-valued.
        Raises if any sample has more than ``max_instances`` objects.
        """
        b = self.batch_size
        k = max_instances
        device = self.images.device

        if b > 0:
            box_dtype = self.boxes[0].as_subclass(torch.Tensor).dtype
            label_dtype = self.labels[0].dtype
        else:
            box_dtype = torch.float32
            label_dtype = torch.int64

        boxes_padded = torch.zeros((b, k, 4), dtype=box_dtype, device=device)
        labels_padded = torch.zeros((b, k), dtype=label_dtype, device=device)
        instance_valid = torch.zeros((b, k), dtype=torch.bool, device=device)

        for i in range(b):
            n = self.boxes[i].size(0)
            if n > k:
                raise ValueError(
                    f"sample {i} has {n} instances, exceeds max_instances={k}"
                )
            if n > 0:
                boxes_padded[i, :n] = self.boxes[i].as_subclass(torch.Tensor)
                labels_padded[i, :n] = self.labels[i]
                instance_valid[i, :n] = True

        instance_masks_padded: torch.Tensor | None = None
        instance_ids_padded: torch.Tensor | None = None
        if self.instance_masks is not None and self.instance_ids is not None:
            h, w = self.images.shape[-2:]
            instance_masks_padded = torch.zeros(
                (b, k, h, w), dtype=torch.bool, device=device
            )
            instance_ids_padded = torch.zeros((b, k), dtype=torch.int32, device=device)
            for i in range(b):
                n = self.instance_masks[i].size(0)
                if n > 0:
                    instance_masks_padded[i, :n] = self.instance_masks[i].as_subclass(
                        torch.Tensor
                    )
                    instance_ids_padded[i, :n] = self.instance_ids[i]

        camera_intrinsics_tensor: torch.Tensor | None = None
        if self.camera_intrinsics is not None:
            camera_intrinsics_tensor = torch.tensor(
                [[c.fx, c.fy, c.cx, c.cy] for c in self.camera_intrinsics],
                dtype=torch.float32,
                device=device,
            )

        return PaddedBatchedDenseSample(
            images=self.images,
            boxes=boxes_padded,
            labels=labels_padded,
            instance_valid=instance_valid,
            max_instances=k,
            instance_masks=instance_masks_padded,
            instance_ids=instance_ids_padded,
            semantic_maps=self.semantic_maps,
            panoptic_maps=self.panoptic_maps,
            depth=self.depth,
            depth_valid=self.depth_valid,
            normals=self.normals,
            padding_mask=self.padding_mask,
            camera_intrinsics=camera_intrinsics_tensor,
        )

    @staticmethod
    def from_padded(padded: PaddedBatchedDenseSample) -> "BatchedDenseSample":
        """Unpack a :class:`PaddedBatchedDenseSample` into a ragged batch.

        Uses ``instance_valid`` as the per-sample gather mask. Reconstructs
        ``tv_tensors.BoundingBoxes`` in XYXY format (the DenseSample canonical
        convention) and unpacks the ``[B, 4]`` intrinsics tensor back into
        :class:`CameraIntrinsics` instances.
        """
        b = padded.batch_size
        h, w = padded.images.shape[-2:]

        boxes: list[tv_tensors.BoundingBoxes] = []
        labels: list[torch.Tensor] = []
        for i in range(b):
            mask = padded.instance_valid[i]
            boxes.append(
                tv_tensors.BoundingBoxes(  # pyright: ignore[reportCallIssue]
                    padded.boxes[i][mask],
                    format=tv_tensors.BoundingBoxFormat.XYXY,
                    canvas_size=(h, w),
                )
            )
            labels.append(padded.labels[i][mask])

        instance_masks: list[InstanceMask] | None = None
        instance_ids: list[torch.Tensor] | None = None
        if padded.instance_masks is not None and padded.instance_ids is not None:
            instance_masks = []
            instance_ids = []
            for i in range(b):
                mask = padded.instance_valid[i]
                instance_masks.append(InstanceMask(padded.instance_masks[i][mask]))
                instance_ids.append(padded.instance_ids[i][mask])

        camera_intrinsics: list[CameraIntrinsics] | None = None
        if padded.camera_intrinsics is not None:
            rows = cast(list[list[float]], padded.camera_intrinsics.tolist())
            camera_intrinsics = [
                CameraIntrinsics(fx=row[0], fy=row[1], cx=row[2], cy=row[3])
                for row in rows
            ]

        return BatchedDenseSample(
            images=padded.images,
            boxes=boxes,
            labels=labels,
            instance_masks=instance_masks,
            instance_ids=instance_ids,
            semantic_maps=padded.semantic_maps,
            panoptic_maps=padded.panoptic_maps,
            depth=padded.depth,
            depth_valid=padded.depth_valid,
            normals=padded.normals,
            padding_mask=padded.padding_mask,
            camera_intrinsics=camera_intrinsics,
        )

    def to_samples(self) -> list[DenseSample]:
        """Unstack back into per-sample :class:`DenseSample` objects."""
        images = self.images.as_subclass(torch.Tensor)
        out: list[DenseSample] = []
        for i in range(self.batch_size):
            fields_dict: dict[str, Any] = {
                "image": tv_tensors.Image(images[i]),
                "boxes": self.boxes[i],
                "labels": self.labels[i],
            }
            if self.instance_masks is not None and self.instance_ids is not None:
                fields_dict["instance_masks"] = self.instance_masks[i]
                fields_dict["instance_ids"] = self.instance_ids[i]
            if self.semantic_maps is not None:
                fields_dict["semantic_map"] = SemanticMap(self.semantic_maps[i])
            if self.panoptic_maps is not None:
                fields_dict["panoptic_map"] = PanopticMap(self.panoptic_maps[i])
            if self.depth is not None:
                fields_dict["depth"] = self.depth[i]
            if self.depth_valid is not None:
                fields_dict["depth_valid"] = self.depth_valid[i]
            if self.normals is not None:
                fields_dict["normals"] = self.normals[i]
            if self.padding_mask is not None:
                fields_dict["padding_mask"] = PaddingMask(self.padding_mask[i])
            if self.camera_intrinsics is not None:
                fields_dict["camera_intrinsics"] = self.camera_intrinsics[i]
            out.append(DenseSample(**fields_dict))
        return out


def _empty_batch() -> BatchedDenseSample:
    """Zero-sized batch used for ``from_samples([])``.

    The ``(0, 3, 0, 0)`` image shape is a placeholder — consumers must gate on
    ``batch_size`` before reading ``images``.
    """
    empty_images = tv_tensors.Image(torch.empty((0, 3, 0, 0), dtype=torch.float32))
    return BatchedDenseSample(images=empty_images, boxes=[], labels=[])


def _stack_optional(
    samples: list[DenseSample],
    field_name: str,
    wrapper: type[Any] | None = None,
) -> Any | None:
    """Stack a per-sample optional tensor field into a batched tensor.

    Returns ``None`` when the field is ``None`` on all samples; raises when the
    field is present on some samples but absent on others (inconsistent modality
    set — ``from_samples`` already checks this, so the raise is a defense).
    """
    values = [getattr(s, field_name) for s in samples]
    none_count = sum(1 for v in values if v is None)
    if none_count == len(values):
        return None
    if none_count > 0:
        raise ValueError(f"{field_name} must be set on all samples or none")
    stacked = torch.stack([v.as_subclass(torch.Tensor) for v in values])
    return wrapper(stacked) if wrapper is not None else stacked
