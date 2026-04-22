"""Property-based tests for copy-paste augmentation functionality."""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from segpaste.augmentation import CopyPasteAugmentation
from segpaste.config import CopyPasteConfig
from segpaste.types import DenseSample, Modality
from tests.strategies import dense_sample_strategy


def _instance_sample_strategy() -> st.SearchStrategy[DenseSample]:
    return dense_sample_strategy({Modality.INSTANCE})


class TestCopyPasteFuzzing:
    """Property-based tests for CopyPasteAugmentation."""

    @settings(deadline=None, max_examples=50)
    @given(
        target=_instance_sample_strategy(),
        source_objects=st.lists(_instance_sample_strategy(), min_size=0, max_size=3),
        paste_prob=st.floats(min_value=0.0, max_value=1.0),
    )
    def test_transform_properties(
        self,
        target: DenseSample,
        source_objects: list[DenseSample],
        paste_prob: float,
    ) -> None:
        """Fuzz the transform method."""
        config = CopyPasteConfig(
            paste_probability=paste_prob,
            max_paste_objects=5,
            min_paste_objects=1,
            min_object_area=0,
        )
        augmentation = CopyPasteAugmentation(config)

        try:
            result = augmentation.transform(target, source_objects)
        except Exception as e:
            pytest.fail(f"Transform failed with: {e}")

        # Properties to check:
        # 1. Output is a valid DenseSample with INSTANCE modality preserved
        assert isinstance(result, DenseSample)
        assert result.instance_masks is not None
        assert result.instance_ids is not None

        # 2. Dimensions consistency
        c, h, w = target.image.shape
        assert result.image.shape == (c, h, w)
        assert (
            result.instance_masks.shape[1:] == (h, w)
            or result.instance_masks.shape[0] == 0
        )

        # 3. Number of items consistency
        num_objs = result.boxes.shape[0]
        assert result.labels.shape[0] == num_objs
        assert result.instance_masks.shape[0] == num_objs
        assert result.instance_ids.shape[0] == num_objs

        # 4. Data types
        assert result.image.dtype == target.image.dtype
        assert result.boxes.dtype == target.boxes.dtype
        assert result.labels.dtype == target.labels.dtype
        assert result.instance_masks.dtype == torch.bool
        assert result.instance_ids.dtype == torch.int32

    @settings(deadline=None, max_examples=20)
    @given(target=_instance_sample_strategy())
    def test_transform_empty_sources(self, target: DenseSample) -> None:
        """Test with empty source objects list."""
        config = CopyPasteConfig(paste_probability=1.0)
        augmentation = CopyPasteAugmentation(config)

        result = augmentation.transform(target, [])

        assert result is target

    @settings(deadline=None, max_examples=20)
    @given(
        target=_instance_sample_strategy(),
        source_objects=st.lists(_instance_sample_strategy(), min_size=1, max_size=3),
    )
    def test_transform_prob_zero(
        self, target: DenseSample, source_objects: list[DenseSample]
    ) -> None:
        """Test with probability 0."""
        config = CopyPasteConfig(paste_probability=0.0)
        augmentation = CopyPasteAugmentation(config)

        result = augmentation.transform(target, source_objects)

        assert result is target

    @settings(deadline=None, max_examples=20)
    @given(
        target=_instance_sample_strategy(),
        source_objects=st.lists(_instance_sample_strategy(), min_size=1, max_size=3),
    )
    def test_idempotence_check(
        self, target: DenseSample, source_objects: list[DenseSample]
    ) -> None:
        """Copy-paste is not idempotent, but same seed → same result."""
        config = CopyPasteConfig(paste_probability=1.0)
        augmentation = CopyPasteAugmentation(config)

        rng_state = torch.get_rng_state()
        import random

        py_rng_state = random.getstate()

        result1 = augmentation.transform(target, source_objects)

        torch.set_rng_state(rng_state)
        random.setstate(py_rng_state)

        result2 = augmentation.transform(target, source_objects)

        assert torch.equal(
            result1.image.as_subclass(torch.Tensor),
            result2.image.as_subclass(torch.Tensor),
        )
        assert torch.equal(
            result1.boxes.as_subclass(torch.Tensor),
            result2.boxes.as_subclass(torch.Tensor),
        )
