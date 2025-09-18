"""Blending operations for copy-paste augmentation."""

from typing import Any

import torch
import torch.nn.functional as F


def alpha_blend(
    source: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, alpha: float = 1.0
) -> torch.Tensor:
    """Apply alpha blending to combine source and target images.

    Args:
        source: Source image tensor of shape [C, H, W]
        target: Target image tensor of shape [C, H, W]
        mask: Binary mask tensor of shape [H, W] or [1, H, W]
        alpha: Alpha value for blending (0.0 to 1.0)

    Returns:
        Blended image tensor of shape [C, H, W]
    """
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)  # Add channel dimension

    # Apply alpha blending
    blended: torch.Tensor = target * (1.0 - alpha * mask) + source * (alpha * mask)

    return blended


def gaussian_blend(
    source: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    sigma: float = 1.0,
    kernel_size: int = 5,
) -> torch.Tensor:
    """Apply Gaussian-weighted blending for smoother transitions.

    Args:
        source: Source image tensor of shape [C, H, W]
        target: Target image tensor of shape [C, H, W]
        mask: Binary mask tensor of shape [H, W] or [1, H, W]
        sigma: Standard deviation for Gaussian kernel
        kernel_size: Size of Gaussian kernel (must be odd)

    Returns:
        Blended image tensor of shape [C, H, W]
    """
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)  # Add channel dimension

    # Create Gaussian kernel
    gaussian_kernel = _create_gaussian_kernel(kernel_size, sigma, mask.device)

    # Apply Gaussian blur to mask for smooth transitions
    padding = kernel_size // 2
    blurred_mask = F.conv2d(
        mask.unsqueeze(0), gaussian_kernel.unsqueeze(0).unsqueeze(0), padding=padding
    ).squeeze(0)

    # Normalize blurred mask
    blurred_mask = torch.clamp(blurred_mask, 0.0, 1.0)

    # Ensure mask has same number of channels as images
    if blurred_mask.shape[0] == 1 and source.shape[0] > 1:
        blurred_mask = blurred_mask.repeat(source.shape[0], 1, 1)

    # Apply blending
    blended: torch.Tensor = target * (1.0 - blurred_mask) + source * blurred_mask

    return blended


def _create_gaussian_kernel(
    kernel_size: int, sigma: float, device: torch.device
) -> torch.Tensor:
    """Create a 2D Gaussian kernel.

    Args:
        kernel_size: Size of the kernel (must be odd)
        sigma: Standard deviation
        device: Device to create tensor on

    Returns:
        Gaussian kernel tensor of shape [kernel_size, kernel_size]
    """
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd")

    # Create coordinate grids
    half_size = kernel_size // 2
    x = torch.arange(-half_size, half_size + 1, dtype=torch.float32, device=device)
    y = torch.arange(-half_size, half_size + 1, dtype=torch.float32, device=device)
    x_grid, y_grid = torch.meshgrid(x, y, indexing="ij")

    # Compute Gaussian kernel
    kernel = torch.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()

    return kernel


def create_smooth_mask_border(
    mask: torch.Tensor, border_width: int = 5, sigma: float = 1.5
) -> torch.Tensor:
    """Create smooth borders on mask edges for better blending.

    Args:
        mask: Binary mask tensor of shape [H, W] or [1, H, W]
        border_width: Width of border to smooth
        sigma: Standard deviation for Gaussian smoothing

    Returns:
        Smoothed mask tensor
    """
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)

    # Create distance transform (approximate)
    eroded_mask = mask.clone()

    # Apply erosion multiple times to create border effect
    kernel_size = 3
    padding = kernel_size // 2
    erosion_kernel = torch.ones(1, 1, kernel_size, kernel_size, device=mask.device)

    for _ in range(border_width):
        eroded_mask = F.conv2d(
            eroded_mask.unsqueeze(0), erosion_kernel, padding=padding
        ).squeeze(0)
        eroded_mask = (eroded_mask >= (kernel_size * kernel_size)).float()

    # Create smooth transition
    border_region = mask - eroded_mask

    # Apply Gaussian smoothing to border region
    if sigma > 0:
        kernel_size = int(4 * sigma) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1

        gaussian_kernel = _create_gaussian_kernel(kernel_size, sigma, mask.device)
        padding = kernel_size // 2

        smoothed_border = F.conv2d(
            border_region.unsqueeze(0),
            gaussian_kernel.unsqueeze(0).unsqueeze(0),
            padding=padding,
        ).squeeze(0)

        # Combine eroded mask with smoothed border
        smooth_mask = eroded_mask + smoothed_border
        smooth_mask = torch.clamp(smooth_mask, 0.0, 1.0)
    else:
        smooth_mask = mask

    return smooth_mask


def blend_with_mode(
    source: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    mode: str = "alpha",
    **kwargs: Any,
) -> torch.Tensor:
    """Apply blending with specified mode.

    Args:
        source: Source image tensor of shape [C, H, W]
        target: Target image tensor of shape [C, H, W]
        mask: Binary mask tensor of shape [H, W] or [1, H, W]
        mode: Blending mode ("alpha", "gaussian")
        **kwargs: Additional arguments for specific blending modes

    Returns:
        Blended image tensor of shape [C, H, W]
    """
    if mode == "alpha":
        alpha = kwargs.get("alpha", 1.0)
        return alpha_blend(source, target, mask, alpha)
    elif mode == "gaussian":
        sigma = kwargs.get("sigma", 1.0)
        kernel_size = kwargs.get("kernel_size", 5)
        return gaussian_blend(source, target, mask, sigma, kernel_size)
    raise ValueError(f"Unknown blending mode: {mode}")
