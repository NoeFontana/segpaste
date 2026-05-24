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

See the [getting-started guide](https://NoeFontana.github.io/segpaste/getting-started/)
for a runnable end-to-end example: install → registered preset →
`BatchCopyPaste` → Hugging Face Mask2Former → one training step in under
50 lines. The full docs site at
<https://NoeFontana.github.io/segpaste> covers migration from
torchvision's reference `SimpleCopyPaste` / `mmdet.CopyPaste`, design
principles, and the pinned public API.

The public API is exposed in the `segpaste` namespace. It is subject to
breaking changes, without prior notice, until version 1.0.0 (see ADR-0003).

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
