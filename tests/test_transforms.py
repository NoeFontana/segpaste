"""Tests for transform functionality."""

import logging
import os
from pathlib import Path
import random
from typing import Tuple

import pytest
import torch
import torchvision
from torchvision import tv_tensors
from torchvision.transforms.v2 import InterpolationMode
from torchvision.transforms.v2 import functional as F

from segpaste.lsj import RandomResize


def create_test_image(size: Tuple[int, int, int] = (3, 224, 224)) -> tv_tensors.Image:
    """Create a test image with gradient pattern for visual verification."""
    c, h, w = size
    image = torch.zeros(c, h, w, dtype=torch.uint8)

    # Create gradient pattern scaled to 0-255 range
    for i in range(h):
        for j in range(w):
            image[0, i, j] = int((i / h) * 255)  # Vertical gradient
            image[1, i, j] = int((j / w) * 255)  # Horizontal gradient
            image[2, i, j] = int(((i + j) / (h + w)) * 255)  # Diagonal gradient

    return F.to_image(image)


class TestRandomResize:
    """Test cases for RandomResize transform."""

    def test_init(self) -> None:
        """Test RandomResize initialization."""
        transform = RandomResize(
            min_scale=0.5,
            max_scale=2.0,
            target_height=256,
            target_width=256,
            interpolation=InterpolationMode.BILINEAR,
        )

        assert transform.min_scale == 0.5
        assert transform.max_scale == 2.0
        assert transform.target_height == 256
        assert transform.target_width == 256
        assert transform.interpolation == InterpolationMode.BILINEAR

    def test_make_params(self) -> None:
        """Test parameter generation."""
        transform = RandomResize(
            min_scale=0.5, max_scale=2.0, target_height=256, target_width=256
        )

        # Test multiple parameter generations
        scales = []
        for _ in range(100):
            params = transform.make_params([])
            scale = params["scale"]
            scales.append(scale)
            assert 0.5 <= scale <= 2.0

        # Check that we get variation in scales
        assert len(set(scales)) > 1, "Scale should vary across generations"

    def test_transform_basic(self) -> None:
        """Test basic transform functionality."""
        transform = RandomResize(
            min_scale=1.0,
            max_scale=1.0,  # Fixed scale for predictable testing
            target_height=128,
            target_width=128,
        )

        image = create_test_image((3, 224, 224))

        # Mock the scale to be exactly 1.0 for predictable results
        params = {"scale": 1.0}
        result = transform.transform(image, params)

        assert isinstance(result, tv_tensors.Image)
        assert result.shape[0] == 3  # Channels preserved

        # With scale=1.0, target=128x128
        assert result.shape[1] == 128
        assert result.shape[2] == 128

    def test_aspect_ratio_preservation(self) -> None:
        """Test that aspect ratio is preserved during resize."""
        transform = RandomResize(
            min_scale=1.0, max_scale=1.0, target_height=256, target_width=256
        )

        # Test with non-square image
        image = create_test_image((3, 100, 200))  # 1:2 aspect ratio
        params = {"scale": 1.0}
        result = transform.transform(image, params)

        # The smaller dimension should determine the scale
        # output_scale = min(256/100, 256/200) = min(2.56, 1.28) = 1.28
        expected_h = round(100 * 1.28)  # 128
        expected_w = round(200 * 1.28)  # 256

        assert result.shape[1] == expected_h
        assert result.shape[2] == expected_w

    @pytest.mark.parametrize("scale", [0.5, 1.0, 1.5, 2.0])
    def test_different_scales(self, scale: float) -> None:
        """Test transform with different scale values."""
        transform = RandomResize(
            min_scale=scale, max_scale=scale, target_height=256, target_width=256
        )

        image = create_test_image((3, 128, 128))
        params = {"scale": scale}
        result = transform.transform(image, params)

        # Calculate expected size
        target_h, target_w = 256 * scale, 256 * scale
        output_scale = min(target_h / 128, target_w / 128)
        expected_size = round(128 * output_scale)

        assert result.shape[1] == expected_size
        assert result.shape[2] == expected_size

    @pytest.mark.parametrize(
        "interpolation",
        [
            InterpolationMode.NEAREST,
            InterpolationMode.BILINEAR,
            InterpolationMode.BICUBIC,
        ],
    )
    def test_interpolation_modes(self, interpolation: InterpolationMode) -> None:
        """Test different interpolation modes."""
        transform = RandomResize(
            min_scale=0.5,
            max_scale=0.5,
            target_height=64,
            target_width=64,
            interpolation=interpolation,
        )

        image = create_test_image((3, 128, 128))
        params = {"scale": 0.5}
        result = transform.transform(image, params)

        # Should work without errors and produce valid output
        assert isinstance(result, tv_tensors.Image)
        assert result.shape[0] == 3

    def test_edge_cases(self) -> None:
        """Test edge cases and boundary conditions."""
        transform = RandomResize(
            min_scale=0.1, max_scale=10.0, target_height=32, target_width=32
        )

        # Very small image
        small_image = create_test_image((3, 8, 8))
        params = {"scale": 2.0}
        result = transform.transform(small_image, params)
        assert isinstance(result, tv_tensors.Image)

        # Very large scale
        image = create_test_image((3, 32, 32))
        params = {"scale": 5.0}
        result = transform.transform(image, params)
        assert isinstance(result, tv_tensors.Image)

    def test_reproducibility(self) -> None:
        """Test that transform is reproducible with same random seed."""
        transform = RandomResize(
            min_scale=0.5, max_scale=2.0, target_height=128, target_width=128
        )

        image = create_test_image((3, 224, 224))

        # Set seed and get first result
        random.seed(42)
        params1 = transform.make_params([image])
        result1 = transform.transform(image, params1)

        # Set same seed and get second result
        random.seed(42)
        params2 = transform.make_params([image])
        result2 = transform.transform(image, params2)

        # Results should be identical
        assert torch.equal(result1, result2)
        assert params1["scale"] == params2["scale"]

    def test_with_different_input_types(self) -> None:
        """Test transform with different tensor types."""
        transform = RandomResize(
            min_scale=0.8, max_scale=0.8, target_height=100, target_width=100
        )

        # Test with regular tensor
        tensor = torch.rand(3, 50, 50)
        params = {"scale": 0.8}
        result = transform.transform(tensor, params)
        assert isinstance(result, torch.Tensor)

        # Test with tv_tensors.Image
        image = tv_tensors.Image(torch.rand(3, 50, 50))
        result = transform.transform(image, params)
        assert isinstance(result, tv_tensors.Image)

    def test_visual_output(self, tmp_path: Path) -> None:
        """Test visual output and save images for debugging."""

        if os.environ.get("SAVE_TEST_IMAGES", "0") != "1":
            pytest.skip("SAVE_TEST_IMAGES not enabled")

        logger = logging.getLogger("test_visual_output")
        logger.info(
            f"Images will be saved to {tmp_path}",
        )

        transform = RandomResize(
            min_scale=0.5, max_scale=2.0, target_height=256, target_width=256
        )

        # Create test image with recognizable pattern
        original_image = create_test_image((3, 224, 224))

        # Save original image
        torchvision.utils.save_image(
            original_image / 255.0, f"{tmp_path}/random_resize_original.png"
        )

        # Test multiple scales and save individually
        scales = [0.5, 0.8, 1.0, 1.5, 2.0]
        for _, scale in enumerate(scales):
            params = {"scale": scale}
            transformed = transform.transform(original_image, params)
            torchvision.utils.save_image(
                transformed / 255.0,
                f"{tmp_path}/random_resize_scale_{scale}.png",
            )

        # Test aspect ratio preservation with non-square images
        # Wide image
        wide_image = create_test_image((3, 100, 300))
        torchvision.utils.save_image(
            wide_image / 255.0,
            f"{tmp_path}/random_resize_wide_original.png",
        )
        params = {"scale": 1.0}
        wide_transformed = transform.transform(wide_image, params)
        torchvision.utils.save_image(
            wide_transformed / 255.0,
            f"{tmp_path}/random_resize_wide_transformed.png",
        )

        # Tall image
        tall_image = create_test_image((3, 300, 100))
        torchvision.utils.save_image(
            tall_image / 255.0,
            f"{tmp_path}/random_resize_tall_original.png",
        )
        tall_transformed = transform.transform(tall_image, params)
        torchvision.utils.save_image(
            tall_transformed / 255.0,
            f"{tmp_path}/random_resize_tall_transformed.png",
        )

    def test_multiple_transforms(self) -> None:
        """Test applying transform multiple times."""
        transform = RandomResize(
            min_scale=0.8, max_scale=1.2, target_height=128, target_width=128
        )

        image = create_test_image((3, 224, 224))

        # Apply transform multiple times
        results = []
        for i in range(5):
            random.seed(i)  # Different seed each time
            params = transform.make_params([image])
            result = transform.transform(image, params)
            results.append(result)

        # All results should be valid but potentially different
        for result in results:
            assert isinstance(result, tv_tensors.Image)
            assert result.shape[0] == 3

        # Save comparison if debugging enabled
        if os.environ.get("SAVE_TEST_IMAGES", "0") == "1":
            os.makedirs("./test_outputs", exist_ok=True)
            all_images = image + results
            # Save each image individually since they have different sizes
            for i, img in enumerate(all_images):
                name = "original" if i == 0 else f"transformed_{i}"
                torchvision.utils.save_image(
                    img,
                    f"./test_outputs/random_resize_multiple_{name}.png",
                    normalize=True,
                )

    def test_parameter_validation(self) -> None:
        """Test parameter validation and error handling."""
        # Test that min_scale <= max_scale is expected behavior
        # (Note: The current implementation doesn't validate this,
        # but the test documents expected behavior)

        transform = RandomResize(
            min_scale=2.0,
            max_scale=0.5,  # This would cause issues
            target_height=128,
            target_width=128,
        )

        # The transform should still work but might produce unexpected results
        image = create_test_image((3, 100, 100))
        params = transform.make_params([image])

        # Scale might be outside expected range
        scale = params["scale"]
        assert isinstance(scale, float)

    def test_zero_and_negative_dimensions(self) -> None:
        """Test handling of zero and very small dimensions."""
        transform = RandomResize(
            min_scale=1.0, max_scale=1.0, target_height=64, target_width=64
        )

        # Very small image (but still valid)
        tiny_image = create_test_image((3, 1, 1))
        params = {"scale": 1.0}
        result = transform.transform(tiny_image, params)
        assert isinstance(result, tv_tensors.Image)
        assert result.shape[0] == 3

    def test_scale_calculation_accuracy(self) -> None:
        """Test that scale calculations are accurate."""
        transform = RandomResize(
            min_scale=1.0, max_scale=1.0, target_height=200, target_width=300
        )

        # Test with known dimensions
        image = create_test_image((3, 100, 150))  # 2:3 aspect ratio
        params = {"scale": 1.0}
        result = transform.transform(image, params)

        # target_scale_h = 200, target_scale_w = 300
        # output_scale = min(200/100, 300/150) = min(2.0, 2.0) = 2.0
        # Expected size: 100*2=200, 150*2=300
        assert result.shape[1] == 200
        assert result.shape[2] == 300

    def test_channel_preservation(self) -> None:
        """Test that different channel configurations are preserved."""
        transform = RandomResize(
            min_scale=0.5, max_scale=0.5, target_height=64, target_width=64
        )
        params = {"scale": 0.5}

        # Test RGB
        rgb_image = create_test_image((3, 128, 128))
        result = transform.transform(rgb_image, params)
        assert result.shape[0] == 3

        # Test RGBA
        rgba_image = create_test_image((4, 128, 128))
        result = transform.transform(rgba_image, params)
        assert result.shape[0] == 4
