"""Property-based tests for copy-paste augmentation functionality."""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from segpaste.augmentation import CopyPasteAugmentation
from segpaste.config import CopyPasteConfig
from segpaste.types import DetectionTarget, Modality
from tests.strategies import dense_sample_strategy


@st.composite
def detection_target_strategy(draw: st.DrawFn) -> DetectionTarget:
    """Draw a :class:`DetectionTarget` via the DenseSample bridge."""
    sample = draw(dense_sample_strategy({Modality.INSTANCE}))
    return sample.to_detection_target()


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
