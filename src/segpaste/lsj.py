import random
from typing import Any, Tuple, Union

import torch
from torchvision import tv_tensors
from torchvision.transforms.v2 import (
    Compose,
    Transform,
    query_size,
)
from torchvision.transforms.v2 import functional as F


class RandomResize(Transform):
    def __init__(
        self,
        min_scale: float,
        max_scale: float,
        target_height: int,
        target_width: int,
    ) -> None:
        """Randomly resize the input image while preserving aspect ratio.

        The final size is obtained by scaling the target height and width with a random
        factor.
        """

        super().__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.target_height = target_height
        self.target_width = target_width

    def make_params(
        self,
        flat_inputs: list[tv_tensors.TVTensor | torch.Tensor],  # noqa: ARG002
    ) -> dict[str, float]:
        scale = random.uniform(self.min_scale, self.max_scale)
        return {"scale": scale}

    def transform(
        self, inpt: tv_tensors.TVTensor | torch.Tensor, params: dict[str, float]
    ) -> Any:
        h, w = F.get_size(inpt)
        scale: float = params["scale"]

        target_scale_h, target_scale_w = (
            self.target_height * scale,
            self.target_width * scale,
        )
        output_scale = min(target_scale_h / h, target_scale_w / w)

        new_h, new_w = round(h * output_scale), round(w * output_scale)
        return self._call_kernel(F.resize, inpt, [new_h, new_w])


class FixedSizeCrop(Transform):
    def __init__(
        self,
        output_height: int,
        output_width: int,
        img_pad_value: float | int = 0,
        seg_pad_value: int = 255,
    ) -> None:
        """Crops the given image to a fixed size.

        Args:
            output_height (int): Desired output height.
            output_width (int): Desired output width.
        """
        super().__init__()
        self.output_height = output_height
        self.output_width = output_width

        self.img_pad_value = img_pad_value
        self.seg_pad_value = seg_pad_value

    def make_params(
        self, flat_inputs: list[tv_tensors.TVTensor | torch.Tensor]
    ) -> dict[str, int]:
        inpt_h, inpt_w = query_size(flat_inputs)

        offset_top = round(random.randint(0, max(0, inpt_h - self.output_height)))
        offset_left = round(random.randint(0, max(0, inpt_w - self.output_width)))

        return {"offset_top": offset_top, "offset_left": offset_left}

    def transform(
        self, inpt: tv_tensors.TVTensor | torch.Tensor, params: dict[str, int]
    ) -> Any:
        h, w = F.get_size(inpt)
        cropped = self._call_kernel(
            F.crop,
            inpt,
            top=params["offset_top"],
            left=params["offset_left"],
            height=self.output_height,
            width=self.output_width,
        )
        # If the input is smaller than the output size, we pad it
        if h < self.output_height:
            if isinstance(cropped, (tv_tensors.Image, tv_tensors.Video)):
                cropped[..., h:, :] = self.img_pad_value
            elif isinstance(cropped, tv_tensors.Mask):
                cropped[..., h:, :] = self.seg_pad_value
        if w < self.output_width:
            if isinstance(cropped, (tv_tensors.Image, tv_tensors.Video)):
                cropped[..., :, w:] = self.img_pad_value
            elif isinstance(cropped, tv_tensors.Mask):
                cropped[..., :, w:] = self.seg_pad_value

        return cropped


def make_large_scale_jittering(
    output_size: Union[int, Tuple[int, int]],
    min_scale: float = 0.1,
    max_scale: float = 2.0,
    img_pad_value: Union[float, int] = 0,
    seg_pad_value: int = 255,
) -> Transform:
    """
    Factory function to create a LargeScaleJittering transform.

    Args:
        output_size (int or tuple): The desired output size (height, width) of the crop.
        min_scale (float): The minimum scale factor for resizing.
        max_scale (float): The maximum scale factor for resizing.
        img_pad_value (float or int): Fill value for image padding.
        seg_pad_value (int): Fill value for segmentation mask padding.

    Returns:
        A Compose transform implementing Large Scale Jittering.
    """
    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    output_height, output_width = output_size

    return Compose(
        [
            RandomResize(
                min_scale=min_scale,
                max_scale=max_scale,
                target_height=output_height,
                target_width=output_width,
            ),
            FixedSizeCrop(
                output_height=output_height,
                output_width=output_width,
                img_pad_value=img_pad_value,
                seg_pad_value=seg_pad_value,
            ),
        ]
    )
