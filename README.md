# segpaste

A PyTorch implementation of "Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation" ([arXiv:2012.07177](https://arxiv.org/abs/2012.07177)).

This package also provides integration with `torchvision` ecosystem.

## Installation

```bash
pip install segpaste
```

Or install from source:

```bash
git clone https://github.com/NoeFontana/segpaste.git
cd segpaste
pip install -e .
```

## Usage

### TorchVision Integration

Convenience types and wrappers are provided to ease integration with `torchvision` datasets and transforms.

- `PaddingMask`: A TVTensor representing a padding mask.
- `CocoDetectionV2`: A CocoDetection dataset that presents an interface compatible with `torchvision.transforms.v2` and with support for padding masks.
- `SanitizeBoundingBoxes`: A small wrapper around `torchvision.transforms.v2.SanitizeBoundingBoxes` that adds support for `PaddingMask`.

### Batched augmentation

`BatchCopyPaste` is the public entry point: an `nn.Module` that consumes a
`PaddedBatchedDenseSample` and returns one. It is graph-compilable under
`torch.compile(fullgraph=True)` and replaces the pre-v0.3.0 CPU collator
plus all four modality-specific wrappers (instance, panoptic, depth-aware,
classmix) with a single GPU-resident pipeline.

```python
import torch
from torch.utils.data import DataLoader

from segpaste import BatchCopyPaste, BatchedDenseSample

augment = BatchCopyPaste()
loader = DataLoader(dataset, batch_size=8, collate_fn=list)

for samples in loader:
    padded = BatchedDenseSample.from_samples(samples).to_padded(max_instances=32)
    padded = augment(padded, generator=torch.Generator(device=padded.images.device))
    # padded is a PaddedBatchedDenseSample ready to feed your model
```

Usage examples live alongside the test suite (`tests/test_batch_copy_paste_shape.py`).

### Further

The public API is exposed in the `segpaste` namespace. It is subject to breaking changes, without prior notice, until version 1.0.0.

## Development

This project uses [Ruff](https://github.com/astral-sh/ruff) for linting/formatting and [pyright](https://github.com/microsoft/pyright) for type checking.

```bash
# Install development dependencies
pip install -e ".[dev,coco]"

# Format, lint, and type-check
ruff format . && ruff check --fix . && pyright

# Run tests
pytest
```

## Contributing

Contributions are welcome. Please open an issue to discuss major changes or submit a pull request.

## Citation

If you use this implementation in your research, please consider citing the original paper:

```bibtex
@article{ghiasi2020simple,
  title={Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation},
  author={Ghiasi, Golnaz and Cui, Yin and Srinivas, Aravind and Qian, Rui and Lin, Tsung-Yi and Cubuk, Ekin D and Le, Quoc V and Zoph, Barret},
  journal={arXiv preprint arXiv:2012.07177},
  year={2020}
}
```

## License

This project is licensed under the MIT License.
