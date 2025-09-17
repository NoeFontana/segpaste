# segpaste

This package is a torch-only reimplementation of the copy-paste augmentation: https://arxiv.org/abs/2012.07177.

## Installation

### Code formatting, linting and type checking

This project uses [Ruff](https://github.com/astral-sh/ruff) for linting and formatting and [mypy](http://mypy-lang.org/) for type checking.

```bash
# Format code
ruff format src tests

# Run linter with auto-fix
ruff check --fix src tests

# Run type checking
mypy src tests
```

## License

MIT License - see LICENSE file for details.
