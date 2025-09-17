.PHONY: help install test format clean

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package
	uv sync --no-dev

test:  ## Run tests
	pytest

test-cov:  ## Run tests with coverage
	pytest --cov=segpaste --cov-report=html --cov-report=term-missing

format:  ## Lint and format code
	uv run ruff check --fix src tests
	uv run ruff format src tests

clean:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

