"""Property-based tests for copy-paste augmentation functionality."""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as npst

from segpaste.augmentation import CopyPasteAugmentation
from segpaste.config import CopyPasteConfig
from segpaste.types import DetectionTarget


# Strategies for generating random tensors and detection targets
@st.composite
def image_tensor_strategy(draw: st.DrawFn) -> torch.Tensor:
    """Generate a random image tensor (C, H, W)."""
    # Restrict sizes to avoid OOM and speed up tests
    h = draw(st.integers(min_value=16, max_value=256))
    w = draw(st.integers(min_value=16, max_value=256))
    c = 3  # Standard RGB

    # Generate numpy array and convert to tensor
    data = draw(
        npst.arrays(
            dtype=float,
            shape=(c, h, w),
            elements=st.floats(min_value=0.0, max_value=1.0),
        )
    )
    return torch.from_numpy(data).float()


@st.composite
def detection_target_strategy(
    draw: st.DrawFn, image_st: st.SearchStrategy[torch.Tensor] | None = None
) -> DetectionTarget:
    """Generate a random DetectionTarget."""
    image = draw(image_tensor_strategy()) if image_st is None else draw(image_st)

    _, h, w = image.shape

    num_objects = draw(st.integers(min_value=0, max_value=5))

    if num_objects == 0:
        return DetectionTarget(
            image=image,
            boxes=torch.zeros((0, 4), dtype=torch.float32),
            labels=torch.zeros((0,), dtype=torch.int64),
            masks=torch.zeros((0, h, w), dtype=torch.float32),
        )

    # Generate boxes
    boxes_list = []
    masks_list = []
    labels_list = []

    for _ in range(num_objects):
        # Generate valid box coordinates
        x1 = draw(st.integers(min_value=0, max_value=w - 2))
        y1 = draw(st.integers(min_value=0, max_value=h - 2))
        x2 = draw(st.integers(min_value=x1 + 1, max_value=w - 1))
        y2 = draw(st.integers(min_value=y1 + 1, max_value=h - 1))

        boxes_list.append([x1, y1, x2, y2])

        # Generate mask for the object
        mask = torch.zeros((h, w), dtype=torch.float32)
        # Fill a random sub-rectangle inside the box to be the mask
        # This ensures mask is inside the box
        mx1 = draw(st.integers(min_value=x1, max_value=x2 - 1))
        my1 = draw(st.integers(min_value=y1, max_value=y2 - 1))
        mx2 = draw(st.integers(min_value=mx1 + 1, max_value=x2))
        my2 = draw(st.integers(min_value=my1 + 1, max_value=y2))

        mask[my1:my2, mx1:mx2] = 1.0
        masks_list.append(mask)

        labels_list.append(draw(st.integers(min_value=1, max_value=100)))

    boxes = torch.tensor(boxes_list, dtype=torch.float32)
    masks = torch.stack(masks_list)
    labels = torch.tensor(labels_list, dtype=torch.int64)

    return DetectionTarget(
        image=image,
        boxes=boxes,
        labels=labels,
        masks=masks,
    )


class TestCopyPasteFuzzing:
    """Property-based tests for CopyPasteAugmentation."""

    @settings(deadline=None, max_examples=50)
    @given(
        target=detection_target_strategy(),
        source_objects=st.lists(detection_target_strategy(), min_size=0, max_size=3),
        paste_prob=st.floats(min_value=0.0, max_value=1.0),
    )
    def test_transform_properties(
        self,
        target: DetectionTarget,
        source_objects: list[DetectionTarget],
        paste_prob: float,
    ) -> None:
        """Fuzz the transform method."""
        config = CopyPasteConfig(
            paste_probability=paste_prob,
            max_paste_objects=5,
            min_paste_objects=1,
            min_object_area=0,
            # Allow small objects for fuzzing to avoid easy filtering
        )
        augmentation = CopyPasteAugmentation(config)

        try:
            result = augmentation.transform(target, source_objects)
        except Exception as e:
            # We want to fail if an unexpected exception occurs
            pytest.fail(f"Transform failed with: {e}")

        # Properties to check:
        # 1. Output is valid DetectionTarget
        assert isinstance(result, DetectionTarget)
        assert isinstance(result.image, torch.Tensor)
        assert isinstance(result.boxes, torch.Tensor)
        assert isinstance(result.labels, torch.Tensor)
        assert isinstance(result.masks, torch.Tensor)

        # 2. Dimensions consistency
        c, h, w = target.image.shape
        assert result.image.shape == (c, h, w)
        assert result.masks.shape[1:] == (h, w) or result.masks.shape[0] == 0

        # 3. Number of items consistency
        num_objs = result.boxes.shape[0]
        assert result.labels.shape[0] == num_objs
        assert result.masks.shape[0] == num_objs

        # 4. Data types
        assert result.image.dtype == target.image.dtype
        assert result.boxes.dtype == target.boxes.dtype  # usually float32
        assert result.labels.dtype == target.labels.dtype
        assert result.masks.dtype == target.masks.dtype

    @settings(deadline=None, max_examples=20)
    @given(
        target=detection_target_strategy(),
    )
    def test_transform_empty_sources(self, target: DetectionTarget) -> None:
        """Test with empty source objects list."""
        config = CopyPasteConfig(paste_probability=1.0)
        augmentation = CopyPasteAugmentation(config)

        result = augmentation.transform(target, [])

        # Should be unchanged
        assert torch.equal(result.image, target.image)
        assert torch.equal(result.boxes, target.boxes)
        assert torch.equal(result.labels, target.labels)
        assert torch.equal(result.masks, target.masks)

    @settings(deadline=None, max_examples=20)
    @given(
        target=detection_target_strategy(),
        source_objects=st.lists(detection_target_strategy(), min_size=1, max_size=3),
    )
    def test_transform_prob_zero(
        self, target: DetectionTarget, source_objects: list[DetectionTarget]
    ) -> None:
        """Test with probability 0."""
        config = CopyPasteConfig(paste_probability=0.0)
        augmentation = CopyPasteAugmentation(config)

        result = augmentation.transform(target, source_objects)

        # Should be unchanged
        assert torch.equal(result.image, target.image)
        assert torch.equal(result.boxes, target.boxes)
        assert torch.equal(result.labels, target.labels)
        assert torch.equal(result.masks, target.masks)

    @settings(deadline=None, max_examples=20)
    @given(
        target=detection_target_strategy(),
        source_objects=st.lists(detection_target_strategy(), min_size=1, max_size=3),
    )
    def test_idempotence_check(
        self, target: DetectionTarget, source_objects: list[DetectionTarget]
    ) -> None:
        """
        Copy-paste is NOT idempotent in general because it adds objects.
        But calling it with same seed should produce same result.
        """
        config = CopyPasteConfig(paste_probability=1.0)
        augmentation = CopyPasteAugmentation(config)

        # Save RNG state
        rng_state = torch.get_rng_state()
        import random

        py_rng_state = random.getstate()

        result1 = augmentation.transform(target, source_objects)

        # Restore RNG state
        torch.set_rng_state(rng_state)
        random.setstate(py_rng_state)

        result2 = augmentation.transform(target, source_objects)

        assert torch.equal(result1.image, result2.image)
        assert torch.equal(result1.boxes, result2.boxes)
