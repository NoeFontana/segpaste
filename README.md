# segpaste

A PyTorch implementation of "Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation" ([arXiv:2012.07177](https://arxiv.org/abs/2012.07177)).

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
