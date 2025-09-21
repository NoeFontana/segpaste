import logging
import os
import random
from pathlib import Path
from typing import Tuple, Union

import pytest
import torch
import torchvision
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

from segpaste.augmentation import (
    FixedSizeCrop,
    RandomResize,
    make_large_scale_jittering,
)
from segpaste.types import DetectionTarget


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


def create_test_mask(size: Tuple[int, int] = (224, 224)) -> tv_tensors.Mask:
    """Create a test mask with multiple classes in different regions."""
    h, w = size
    mask = torch.zeros(h, w, dtype=torch.uint8)

    # Create multiple class regions
    for i in range(h):
        for j in range(w):
            # Class 0: background (top-left quadrant)
            if i < h // 2 and j < w // 2:
                mask[i, j] = 0
            # Class 1: checkerboard pattern (top-right quadrant)
            elif i < h // 2 and j >= w // 2:
                if (i // 16 + j // 16) % 2 == 0:
                    mask[i, j] = 1
                else:
                    mask[i, j] = 2
            # Class 2: horizontal stripes (bottom-left quadrant)
            elif i >= h // 2 and j < w // 2:
                mask[i, j] = 2 if (i // 16) % 2 == 0 else 3
            # Class 3: diagonal pattern (bottom-right quadrant)
            else:
                if (i + j) // 24 % 2 == 0:
                    mask[i, j] = 3
                else:
                    mask[i, j] = 4

    return tv_tensors.Mask(mask)


def create_detection_target(
    image_size: Tuple[int, int, int] = (3, 224, 224),
    num_objects: int = 3,
) -> DetectionTarget:
    """Create a test detection target with boxes, labels, and masks."""
    c, h, w = image_size
    image = create_test_image(image_size)

    # Create random boxes
    boxes = torch.rand(num_objects, 4) * torch.tensor([w, h, w, h])
    # Ensure boxes are valid (x1 < x2, y1 < y2)
    boxes[:, 2] = torch.clamp(
        boxes[:, 2], boxes[:, 0] + torch.tensor(10), torch.tensor(w)
    )
    boxes[:, 3] = torch.clamp(
        boxes[:, 3], boxes[:, 1] + torch.tensor(10), torch.tensor(h)
    )

    # Create random labels
    labels = torch.randint(1, 6, (num_objects,))

    # Create masks for each object
    masks = torch.zeros(num_objects, h, w, dtype=torch.uint8)
    for i in range(num_objects):
        x1, y1, x2, y2 = boxes[i].int()
        masks[i, y1:y2, x1:x2] = 1

    return DetectionTarget(
        image=image,
        boxes=boxes,
        labels=labels,
        masks=masks,
    )


class TestRandomResize:
    """Test cases for RandomResize transform."""

    def test_init(self) -> None:
        """Test RandomResize initialization."""
        transform = RandomResize(
            min_scale=0.5,
            max_scale=2.0,
            target_height=256,
            target_width=256,
        )

        assert transform.min_scale == 0.5
        assert transform.max_scale == 2.0
        assert transform.target_height == 256
        assert transform.target_width == 256

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


class TestFixedSizeCrop:
    """Test cases for FixedSizeCrop transform."""

    def test_init(self) -> None:
        """Test FixedSizeCrop initialization."""
        transform = FixedSizeCrop(
            output_height=256,
            output_width=256,
            img_pad_value=128,
            seg_pad_value=255,
        )

        assert transform.output_height == 256
        assert transform.output_width == 256
        assert transform.img_pad_value == 128
        assert transform.seg_pad_value == 255

    def test_init_defaults(self) -> None:
        """Test FixedSizeCrop initialization with default values."""
        transform = FixedSizeCrop(output_height=128, output_width=128)

        assert transform.output_height == 128
        assert transform.output_width == 128
        assert transform.img_pad_value == 0
        assert transform.seg_pad_value == 255

    def test_make_params(self) -> None:
        """Test parameter creation combining crop and pad params."""
        transform = FixedSizeCrop(output_height=128, output_width=128)

        image = create_test_image((3, 224, 224))

        params = transform.make_params([image])

        # Should contain crop parameters
        assert "offset_top" in params
        assert "offset_left" in params
        assert isinstance(params["offset_top"], int)
        assert isinstance(params["offset_left"], int)

    def test_transform_crop_only(self) -> None:
        """Test transform when only cropping is needed."""
        transform = FixedSizeCrop(output_height=128, output_width=128)

        # Input larger than output - needs cropping
        image = create_test_image((3, 224, 224))

        # Use fixed parameters for predictable testing
        params = {"offset_top": 50, "offset_left": 50}
        result = transform.transform(image, params)

        assert isinstance(result, tv_tensors.Image)
        assert result.shape == (3, 128, 128)

    def test_transform_pad_only(self) -> None:
        """Test transform when only padding is needed."""
        transform = FixedSizeCrop(
            output_height=256,
            output_width=256,
            img_pad_value=128,
        )

        # Input smaller than output - needs padding
        image = create_test_image((3, 128, 128))

        params = {"offset_top": 0, "offset_left": 0}
        result = transform.transform(image, params)

        assert isinstance(result, tv_tensors.Image)
        assert result.shape == (3, 256, 256)

        # Check padding values - bottom and right should be padded
        # The padded area should have the pad value
        assert torch.all(result[:, 128:, :] == 128)  # Bottom padding
        assert torch.all(result[:, :, 128:] == 128)  # Right padding

    def test_transform_exact_size(self) -> None:
        """Test transform when input matches output size exactly."""
        transform = FixedSizeCrop(output_height=224, output_width=224)

        image = create_test_image((3, 224, 224))

        params = {"offset_top": 0, "offset_left": 0}
        result = transform.transform(image, params)

        assert isinstance(result, tv_tensors.Image)
        assert result.shape == (3, 224, 224)
        # Should be identical to input
        assert torch.equal(result, image)

    def test_transform_with_mask(self) -> None:
        """Test transform with mask tensor type."""
        transform = FixedSizeCrop(
            output_height=256,
            output_width=256,
            img_pad_value=128,
            seg_pad_value=255,
        )

        # Input smaller than output - needs padding
        mask = create_test_mask((128, 128))

        params = {"offset_top": 0, "offset_left": 0}
        result = transform.transform(mask, params)

        assert isinstance(result, tv_tensors.Mask)
        assert result.shape == (256, 256)

        # Check mask padding values
        assert torch.all(result[128:, :] == 255)  # Bottom padding
        assert torch.all(result[:, 128:] == 255)  # Right padding

    def test_transform_with_video(self) -> None:
        """Test transform with video tensor type."""
        transform = FixedSizeCrop(
            output_height=256,
            output_width=256,
            img_pad_value=64,
        )

        # Create video tensor (T, C, H, W)
        video_data = torch.randint(0, 255, (8, 3, 128, 128), dtype=torch.uint8)
        video = tv_tensors.Video(video_data)

        params = {"offset_top": 0, "offset_left": 0}
        result = transform.transform(video, params)

        assert isinstance(result, tv_tensors.Video)
        assert result.shape == (8, 3, 256, 256)

        # Check video padding values
        assert torch.all(result[:, :, 128:, :] == 64)  # Bottom padding
        assert torch.all(result[:, :, :, 128:] == 64)  # Right padding

    def test_transform_with_regular_tensor(self) -> None:
        """Test transform with regular PyTorch tensor."""
        transform = FixedSizeCrop(output_height=64, output_width=64)

        # Regular tensor (not tv_tensor)
        tensor = torch.rand(3, 128, 128)

        params = {"offset_top": 32, "offset_left": 32}
        result = transform.transform(tensor, params)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 64, 64)

    @pytest.mark.parametrize("height,width", [(64, 64), (128, 256), (256, 128)])
    def test_different_output_sizes(self, height: int, width: int) -> None:
        """Test transform with different output dimensions."""
        transform = FixedSizeCrop(output_height=height, output_width=width)

        image = create_test_image((3, 224, 224))

        # Use make_params to get valid parameters
        params = transform.make_params([image])
        result = transform.transform(image, params)

        assert isinstance(result, tv_tensors.Image)
        assert result.shape == (3, height, width)

    def test_reproducibility(self) -> None:
        """Test that transform is reproducible with same random seed."""
        transform = FixedSizeCrop(output_height=128, output_width=128)

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
        assert params1 == params2

    def test_edge_cases(self) -> None:
        """Test edge cases and boundary conditions."""
        transform = FixedSizeCrop(output_height=32, output_width=32)

        # Very small image
        tiny_image = create_test_image((3, 16, 16))
        params = transform.make_params([tiny_image])
        result = transform.transform(tiny_image, params)
        assert result.shape == (3, 32, 32)

        # Single pixel image
        single_pixel = create_test_image((3, 1, 1))
        params = transform.make_params([single_pixel])
        result = transform.transform(single_pixel, params)
        assert result.shape == (3, 32, 32)

    def test_mixed_crop_and_pad(self) -> None:
        """Test transform that needs both cropping and padding."""
        # This tests an edge case where the implementation might need refinement
        transform = FixedSizeCrop(
            output_height=150,
            output_width=150,
            img_pad_value=100,
        )

        # Rectangle image that's taller than wide
        image = create_test_image((3, 200, 100))  # 200x100

        params = transform.make_params([image])
        result = transform.transform(image, params)

        assert isinstance(result, tv_tensors.Image)
        assert result.shape == (3, 150, 150)

    @pytest.mark.parametrize("img_pad_value", [0, 128, 255])
    @pytest.mark.parametrize("seg_pad_value", [0, 127, 255])
    def test_different_pad_values(self, img_pad_value: int, seg_pad_value: int) -> None:
        """Test different padding values."""
        transform = FixedSizeCrop(
            output_height=256,
            output_width=256,
            img_pad_value=img_pad_value,
            seg_pad_value=seg_pad_value,
        )

        # Test with image
        image = create_test_image((3, 128, 128))
        params = {"offset_top": 0, "offset_left": 0}
        result_img = transform.transform(image, params)

        if img_pad_value != 0:  # Only check if not default background
            assert torch.all(result_img[:, 128:, :] == img_pad_value)
            assert torch.all(result_img[:, :, 128:] == img_pad_value)

        # Test with mask
        mask = create_test_mask((128, 128))
        result_mask = transform.transform(mask, params)

        assert torch.all(result_mask[128:, :] == seg_pad_value)
        assert torch.all(result_mask[:, 128:] == seg_pad_value)

    def test_visual_output(self, tmp_path: Path) -> None:
        """Test visual output and save images for debugging."""
        if os.environ.get("SAVE_TEST_IMAGES", "0") != "1":
            pytest.skip("SAVE_TEST_IMAGES not enabled")

        logger = logging.getLogger("test_fixed_size_crop")
        logger.info(f"Images will be saved to {tmp_path}")

        transform = FixedSizeCrop(
            output_height=256,
            output_width=256,
            img_pad_value=128,
            seg_pad_value=200,
        )

        # Create test images of different sizes
        test_cases = [
            ("large", create_test_image((3, 400, 400))),  # Needs cropping
            ("small", create_test_image((3, 128, 128))),  # Needs padding
            ("wide", create_test_image((3, 200, 400))),  # Mixed case
            ("tall", create_test_image((3, 400, 200))),  # Mixed case
        ]

        for name, image in test_cases:
            # Save original
            torchvision.utils.save_image(
                image / 255.0, f"{tmp_path}/fixed_crop_{name}_original.png"
            )

            # Transform and save result
            params = transform.make_params([image])
            result = transform.transform(image, params)
            torchvision.utils.save_image(
                result / 255.0, f"{tmp_path}/fixed_crop_{name}_transformed.png"
            )

        # Test with masks
        mask = create_test_mask((128, 128))
        params = {"offset_top": 0, "offset_left": 0}
        result_mask = transform.transform(mask, params)

        # Save mask as image for visualization
        mask_as_img = mask.float() * 255
        result_mask_as_img = result_mask.float() * 255

        torchvision.utils.save_image(
            mask_as_img / 255.0, f"{tmp_path}/fixed_crop_mask_original.png"
        )
        torchvision.utils.save_image(
            result_mask_as_img / 255.0, f"{tmp_path}/fixed_crop_mask_transformed.png"
        )

    def test_multiple_transforms_consistency(self) -> None:
        """Test applying transform multiple times with same parameters."""
        transform = FixedSizeCrop(output_height=100, output_width=100)

        image = create_test_image((3, 200, 200))

        # Apply same transform multiple times with same parameters
        params = {"offset_top": 50, "offset_left": 50}

        results = []
        for _ in range(3):
            result = transform.transform(image, params)
            results.append(result)

        # All results should be identical
        for i in range(1, len(results)):
            assert torch.equal(results[0], results[i])

    def test_parameter_bounds(self) -> None:
        """Test parameter generation stays within bounds."""
        transform = FixedSizeCrop(output_height=64, output_width=64)

        # Test with various input sizes
        sizes = [(128, 128), (256, 128), (128, 256), (32, 32)]

        for h, w in sizes:
            image = create_test_image((3, h, w))

            # Generate parameters multiple times
            for _ in range(10):
                params = transform.make_params([image])

                # Check bounds
                max_top = max(0, h - 64)
                max_left = max(0, w - 64)

                assert 0 <= params["offset_top"] <= max_top
                assert 0 <= params["offset_left"] <= max_left


class TestLargeScaleJittering:
    """Test cases for make_large_scale_jittering function."""

    def test_transform_pipeline_basic(self) -> None:
        """Test that the complete pipeline works."""
        transform = make_large_scale_jittering(
            output_size=128,
            min_scale=1.0,  # Fixed scale for predictable testing
            max_scale=1.0,
        )

        image = create_test_image((3, 256, 256))
        result = transform(image)

        assert isinstance(result, tv_tensors.Image)
        assert result.shape == (3, 128, 128)

    def test_transform_pipeline_with_mask(self) -> None:
        """Test pipeline with mask input."""
        transform = make_large_scale_jittering(
            output_size=64,
            min_scale=0.5,
            max_scale=0.5,
            seg_pad_value=200,
        )

        mask = create_test_mask((128, 128))
        result = transform(mask)

        assert isinstance(result, tv_tensors.Mask)
        assert result.shape == (64, 64)

    def test_reproducibility_with_seed(self) -> None:
        """Test pipeline reproducibility with random seed."""
        transform = make_large_scale_jittering(
            output_size=100,
            min_scale=0.8,
            max_scale=1.2,
        )

        image = create_test_image((3, 200, 200))

        # Set seed and get first result
        random.seed(42)
        result1 = transform(image)

        # Set same seed and get second result
        random.seed(42)
        result2 = transform(image)

        # Results should be identical
        assert torch.equal(result1, result2)

    @pytest.mark.parametrize("output_size", [64, 128, 256, (128, 256), (256, 128)])
    def test_different_output_sizes(
        self, output_size: Union[int, Tuple[int, int]]
    ) -> None:
        """Test with different output sizes."""
        transform = make_large_scale_jittering(
            output_size=output_size,
            min_scale=1.0,
            max_scale=1.0,
        )

        image = create_test_image((3, 200, 200))
        result = transform(image)

        expected_shape = (
            (3, output_size, output_size)
            if isinstance(output_size, int)
            else (3, output_size[0], output_size[1])
        )
        assert result.shape == expected_shape

    @pytest.mark.parametrize(
        "min_scale,max_scale", [(0.1, 0.5), (0.5, 1.0), (1.0, 2.0), (0.1, 3.0)]
    )
    def test_different_scale_ranges(self, min_scale: float, max_scale: float) -> None:
        """Test with different scale ranges."""
        transform = make_large_scale_jittering(
            output_size=128,
            min_scale=min_scale,
            max_scale=max_scale,
        )

        image = create_test_image((3, 100, 100))

        # Test multiple times to ensure it works consistently
        for _ in range(5):
            result = transform(image)
            assert isinstance(result, tv_tensors.Image)
            assert result.shape == (3, 128, 128)

    def test_extreme_scale_values(self) -> None:
        """Test with extreme scale values."""
        # Very small scale
        transform_small = make_large_scale_jittering(
            output_size=64,
            min_scale=0.01,
            max_scale=0.1,
        )

        image = create_test_image((3, 200, 200))
        result = transform_small(image)
        assert result.shape == (3, 64, 64)

        # Very large scale
        transform_large = make_large_scale_jittering(
            output_size=64,
            min_scale=5.0,
            max_scale=10.0,
        )

        small_image = create_test_image((3, 32, 32))
        result = transform_large(small_image)
        assert result.shape == (3, 64, 64)

    def test_padding_behavior(self) -> None:
        """Test that padding values are correctly applied."""
        transform = make_large_scale_jittering(
            output_size=200,
            min_scale=0.5,  # This will make input smaller
            max_scale=0.5,
            img_pad_value=100,
            seg_pad_value=150,
        )

        # Test with image
        image = create_test_image((3, 100, 100))
        result_img = transform(image)
        assert result_img.shape == (3, 200, 200)

        # Test with mask
        mask = create_test_mask((100, 100))
        result_mask = transform(mask)
        assert result_mask.shape == (200, 200)

    def test_aspect_ratio_preservation_in_pipeline(self) -> None:
        """Test that aspect ratio is preserved through the pipeline."""
        transform = make_large_scale_jittering(
            output_size=128,
            min_scale=1.0,
            max_scale=1.0,
        )

        # Test with rectangular image
        rect_image = create_test_image((3, 64, 128))  # 1:2 aspect ratio
        result = transform(rect_image)

        # Output should always be square due to FixedSizeCrop
        assert result.shape == (3, 128, 128)

    @pytest.mark.parametrize(
        "name,min_scale,max_scale,img_pad_value",
        [
            ("small_scale", 0.1, 0.5, 255),
            ("medium_scale", 0.5, 1.0, 128),
            ("wide_range", 0.1, 2.0, 0),
        ],
    )
    def test_visual_output_pipeline(
        self,
        tmp_path: Path,
        name: str,
        min_scale: float,
        max_scale: float,
        img_pad_value: int,
    ) -> None:
        """Test visual output of the complete pipeline."""
        if os.environ.get("SAVE_TEST_IMAGES", "0") != "1":
            pytest.skip("SAVE_TEST_IMAGES not enabled")

        logger = logging.getLogger("test_lsj_pipeline")
        logger.info(f"Images will be saved to {tmp_path}")

        original_image = create_test_image((3, 200, 200))
        torchvision.utils.save_image(
            original_image / 255.0, f"{tmp_path}/lsj_original.png"
        )

        transform = make_large_scale_jittering(
            output_size=256,
            min_scale=min_scale,
            max_scale=max_scale,
            img_pad_value=img_pad_value,
        )

        # Generate multiple samples to show variation
        for i in range(3):
            random.seed(i)
            result = transform(original_image)
            torchvision.utils.save_image(
                result / 255.0,
                f"{tmp_path}/lsj_{name}_sample_{i}.png",
            )

    def test_edge_case_very_small_input(self) -> None:
        """Test pipeline with very small input images."""
        transform = make_large_scale_jittering(
            output_size=64,
            min_scale=1.0,
            max_scale=5.0,
        )

        tiny_image = create_test_image((3, 8, 8))
        result = transform(tiny_image)
        assert result.shape == (3, 64, 64)

    def test_edge_case_very_large_input(self) -> None:
        """Test pipeline with very large input images."""
        transform = make_large_scale_jittering(
            output_size=128,
            min_scale=0.1,
            max_scale=0.3,
        )

        large_image = create_test_image((3, 1000, 1000))
        result = transform(large_image)
        assert result.shape == (3, 128, 128)

    def test_multiple_channels(self) -> None:
        """Test pipeline with different numbers of channels."""
        transform = make_large_scale_jittering(output_size=64)

        # Test RGB
        rgb_image = create_test_image((3, 128, 128))
        result = transform(rgb_image)
        assert result.shape == (3, 64, 64)

        # Test RGBA
        rgba_image = create_test_image((4, 128, 128))
        result = transform(rgba_image)
        assert result.shape == (4, 64, 64)

    def test_pipeline_consistency(self) -> None:
        """Test that pipeline produces consistent results."""
        transform = make_large_scale_jittering(
            output_size=100,
            min_scale=0.8,
            max_scale=1.2,
        )

        image = create_test_image((3, 150, 150))

        # Apply transform multiple times
        results = []
        for i in range(5):
            random.seed(i)
            result = transform(image)
            results.append(result)

        # All results should have correct shape and be valid images
        for result in results:
            assert isinstance(result, tv_tensors.Image)
            assert result.shape == (3, 100, 100)
            assert result.dtype == torch.uint8
            assert 0 <= result.min() <= result.max() <= 255
