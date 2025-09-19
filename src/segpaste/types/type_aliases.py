"""Type aliases for tensor types used throughout the package."""

import torch

# Tensor type aliases for better code readability and type safety
ImageTensor = torch.Tensor  # [C, H, W] - Color channel, Height, Width
BoxesTensor = torch.Tensor  # [N, 4] - N bounding boxes in xyxy format
MasksTensor = torch.Tensor  # [N, H, W] - N binary masks
LabelsTensor = torch.Tensor  # [N] - N class labels
