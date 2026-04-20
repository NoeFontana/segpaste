"""Dense-sample container and companion types."""

from collections.abc import Mapping
from dataclasses import dataclass, fields
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

import torch
from torchvision import tv_tensors
from torchvision.tv_tensors import Mask

from segpaste.compile_util import skip_if_compiling
from segpaste.types.data_structures import DetectionTarget, PaddingMask


class Modality(Enum):
    """Dense-sample modalities. IMAGE is always active; the others gate fields."""

    IMAGE = "image"
    INSTANCE = "instance"
    PANOPTIC = "panoptic"
    SEMANTIC = "semantic"
    DEPTH = "depth"
    NORMALS = "normals"


class InstanceMask(Mask):
    """Per-instance binary masks. Shape [N, H, W], dtype bool."""

    if TYPE_CHECKING:

        def __new__(cls, data: Any, **kwargs: Any) -> "InstanceMask": ...


class SemanticMap(Mask):
    """Per-pixel semantic class ids. Shape [H, W], dtype int64, ignore = 255."""

    if TYPE_CHECKING:

        def __new__(cls, data: Any, **kwargs: Any) -> "SemanticMap": ...


class PanopticMap(Mask):
    """Per-pixel panoptic id encoding. Shape [H, W], dtype int64."""

    if TYPE_CHECKING:

        def __new__(cls, data: Any, **kwargs: Any) -> "PanopticMap": ...


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    """Pinhole camera intrinsics in pixel coordinates.

    Required on a :class:`DenseSample` when any composite is constructed with
    ``metric_depth=True``.
    """

    fx: float
    fy: float
    cx: float
    cy: float


@runtime_checkable
class PanopticSchema(Protocol):
    """Panoptic class taxonomy, passed explicitly at composite construction."""

    classes: Mapping[int, Literal["thing", "stuff"]]
    ignore_index: int
    max_instances_per_image: int


@dataclass(slots=True)
class DenseSample:
    """Canonical per-sample container for dense-label Copy-Paste.

    Modality-specific fields are ``None`` when their modality is not active.
    Use :meth:`active_modalities` to derive the active set.
    """

    image: tv_tensors.Image  # [C, H, W]
    boxes: tv_tensors.BoundingBoxes  # [N, 4], xyxy
    labels: torch.Tensor  # [N], int64
    instance_masks: InstanceMask | None = None  # [N, H, W], bool
    semantic_map: SemanticMap | None = None  # [H, W], int64
    panoptic_map: PanopticMap | None = None  # [H, W], int64
    depth: torch.Tensor | None = None  # [1, H, W], float32
    depth_valid: torch.Tensor | None = None  # [1, H, W], bool
    normals: torch.Tensor | None = None  # [3, H, W], float32
    padding_mask: PaddingMask | None = None  # [1, H, W], bool
    camera_intrinsics: CameraIntrinsics | None = None

    @skip_if_compiling
    def __post_init__(self) -> None:
        h, w = self.image.shape[-2:]

        if self.boxes.size(0) != self.labels.size(0):
            raise ValueError("boxes and labels must have same number of objects")

        if self.instance_masks is not None:
            if self.instance_masks.size(0) != self.boxes.size(0):
                raise ValueError(
                    "instance_masks and boxes must have same number of objects"
                )
            if self.instance_masks.shape[-2:] != (h, w):
                raise ValueError("instance_masks must share H, W with image")

        if self.semantic_map is not None and self.semantic_map.shape[-2:] != (h, w):
            raise ValueError("semantic_map must share H, W with image")

        if self.panoptic_map is not None and self.panoptic_map.shape[-2:] != (h, w):
            raise ValueError("panoptic_map must share H, W with image")

        if self.depth is not None and self.depth.shape[-2:] != (h, w):
            raise ValueError("depth must share H, W with image")

        if self.depth_valid is not None and self.depth_valid.shape[-2:] != (h, w):
            raise ValueError("depth_valid must share H, W with image")

        if self.normals is not None and self.normals.shape[-2:] != (h, w):
            raise ValueError("normals must share H, W with image")

        if self.padding_mask is not None and self.padding_mask.shape[1:] != (h, w):
            raise ValueError("padding_mask must share H, W with image")

        # Depth consistency: both fields together, or neither.
        if (self.depth is None) ^ (self.depth_valid is None):
            raise ValueError("depth and depth_valid must both be set or both be None")

    def active_modalities(self) -> set[Modality]:
        """Return the set of active modalities for this sample."""
        active: set[Modality] = {Modality.IMAGE}
        if self.instance_masks is not None:
            active.add(Modality.INSTANCE)
        if self.semantic_map is not None:
            active.add(Modality.SEMANTIC)
        if self.panoptic_map is not None:
            active.add(Modality.PANOPTIC)
        if self.depth is not None:
            active.add(Modality.DEPTH)
        if self.normals is not None:
            active.add(Modality.NORMALS)
        return active

    def to_dict(self) -> dict[str, Any]:
        """Round-trippable dict representation. Omits ``None`` fields."""
        return {
            f.name: value
            for f in fields(self)
            if (value := getattr(self, f.name)) is not None
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "DenseSample":
        names = {f.name for f in fields(DenseSample)}
        return DenseSample(**{k: v for k, v in data.items() if k in names})

    def to_detection_target(self) -> DetectionTarget:
        """Project to the legacy instance-only container.

        Bridge for the P0.B → P1 transition. Requires :class:`Modality.INSTANCE`
        to be active; raises otherwise.
        """
        if self.instance_masks is None:
            raise ValueError(
                "to_detection_target requires instance_masks (INSTANCE modality)"
            )
        return DetectionTarget(
            image=torch.as_tensor(self.image),
            boxes=torch.as_tensor(self.boxes),
            labels=self.labels,
            masks=torch.as_tensor(self.instance_masks),
            padding_mask=self.padding_mask,
        )

    @staticmethod
    def from_detection_target(target: DetectionTarget) -> "DenseSample":
        """Lift a legacy :class:`DetectionTarget` into a DenseSample."""
        h, w = target.image.shape[-2:]
        image = tv_tensors.Image(target.image)
        boxes = tv_tensors.BoundingBoxes(  # pyright: ignore[reportCallIssue]
            target.boxes,
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=(h, w),
        )
        instance_masks = InstanceMask(target.masks.to(torch.bool))
        return DenseSample(
            image=image,
            boxes=boxes,
            labels=target.labels,
            instance_masks=instance_masks,
            padding_mask=target.padding_mask,
        )
