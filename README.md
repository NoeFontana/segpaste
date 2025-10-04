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

### Collator

The recommended interface is through the `CopyPasteCollator` which can be used in lieu of a standard collate function in a PyTorch `DataLoader` as long as the batch_size is greater than 1.

```python
from segpaste import CopyPasteAugmentation, CopyPasteCollator, CopyPasteConfig

config = CopyPasteConfig()
augmentation = CopyPasteAugmentation(config)
collator = CopyPasteCollator(augmentation)
```

Examples of usage can be found in the test suite.

### Transform (Unstable API)

A minimal working example with `CopyPasteTransform` can be found in the [examples](https://github.com/NoeFontana/segpaste/tree/main/examples).

### Further

The public API is exposed in the `segpaste` namespace. It is subject to breaking changes, without prior notice, until version 1.0.0.

## Development

This project uses [Ruff](https://github.com/astral-sh/ruff) for linting/formatting and [mypy](http://mypy-lang.org/) for type checking.

```bash
# Install development dependencies
pip install -e ".[dev,coco]"

# Format, lint, and type-check
ruff format . && ruff check --fix . && mypy .

# Run tests
pytest
```

## Contributing

Contributions are welcome. Please open an issue to discuss major changes or submit a pull request.

## Citation

If you use this implementation in your research, please consider citing the original paper:

```bibtex
@inproceedings{ghiasi2021simple,
  title={Simple copy-paste is a strong data augmentation method for instance segmentation},
  author={Ghiasi, Golnaz and Cui, Yin and Srinivas, Aravind and Qian, Rui and Lin, Tsung-Yi and Cubuk, Ekin D and Le, Quoc V and Zoph, Barret},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2881--2890},
  year={2021}
}
```

## License

This project is licensed under the MIT License.
