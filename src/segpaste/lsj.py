import random
from typing import Any

import torch
from torchvision import tv_tensors
from torchvision.transforms.v2 import (
    InterpolationMode,
    Transform,
)
from torchvision.transforms.v2 import functional as F


class RandomResize(Transform):
    def __init__(
        self,
        min_scale: float,
        max_scale: float,
        target_height: int,
        target_width: int,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
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

        self.interpolation = interpolation

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
        return self._call_kernel(
            F.resize, inpt, [new_h, new_w], interpolation=self.interpolation
        )
