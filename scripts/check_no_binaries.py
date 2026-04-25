"""Binary-file CI guard (ADR-0009 §4).

Walks the working tree (or a list of paths passed as argv) and rejects:

- any path matching an image extension (``*.png``, ``*.jpg``, ``*.jpeg``,
  ``*.tiff``, ``*.tif``, ``*.webp``, ``*.bmp``, ``*.gif``);
- any model / tensor file (``*.pt``, ``*.pth``, ``*.ckpt``,
  ``*.safetensors``) outside ``tests/fixtures/``.

Exit codes
----------
* ``0`` — no violations.
* ``1`` — one or more violations; each is printed as a one-line message
  naming the offending path and the rule that triggered.

The repo holds pure code, ADR text, and synthetic fixtures only. This
script is the mechanical enforcement of ADR-0009's in-repo / out-of-repo
boundary; it is not a style check, it is a redistribution-license guard.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

_IMAGE_SUFFIXES = frozenset(
    {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".webp", ".bmp", ".gif"}
)
_TENSOR_SUFFIXES = frozenset({".pt", ".pth", ".ckpt", ".safetensors"})

# Tensor files are allowed under this prefix: synthetic Hypothesis-derived
# fixtures live in ``tests/fixtures/synthetic/`` and the kernel-regression
# ``ks_snapshot.pt`` (ADR-0008 §D6) lives at ``tests/fixtures/``. Both are
# checked-in test artifacts, not derived from licensed real data.
_TENSOR_ALLOWLIST_PREFIX = "tests/fixtures/"

# Directories pruned during the repo walk: build / cache artifacts and the
# venv. ``os.walk`` is given the chance to skip these in-place so a 60k-file
# ``.venv`` doesn't dominate the run.
_WALK_EXCLUDES = frozenset(
    {
        ".git",
        ".venv",
        ".ruff_cache",
        "__pycache__",
        "htmlcov",
        "ks_report",
        "site",
        "dist",
        "build",
        "node_modules",
        "local_gallery",
    }
)


def _iter_repo_files(root: Path) -> Iterator[Path]:
    """Yield every file under *root* with cache / venv directories pruned."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _WALK_EXCLUDES]
        for name in filenames:
            yield Path(dirpath) / name


def check_paths(paths: Iterable[Path], root: Path) -> list[str]:
    """Return a list of one-line violation messages; empty means pass."""
    violations: list[str] = []
    for path in paths:
        suffix = path.suffix.lower()
        try:
            rel = path.relative_to(root)
        except ValueError:
            rel = path
        if suffix in _IMAGE_SUFFIXES:
            violations.append(f"{rel}: image files are forbidden in-repo (ADR-0009 §4)")
            continue
        if suffix in _TENSOR_SUFFIXES and not rel.as_posix().startswith(
            _TENSOR_ALLOWLIST_PREFIX
        ):
            violations.append(
                f"{rel}: tensor / checkpoint files allowed only under "
                f"{_TENSOR_ALLOWLIST_PREFIX} (ADR-0009 §4)"
            )
    return violations


def _resolve_argv_paths(paths: list[Path]) -> list[Path]:
    """Resolve argv-supplied paths; raise on anything that isn't a file."""
    resolved: list[Path] = []
    for raw in paths:
        path = raw if raw.is_absolute() else raw.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"not a file: {raw}")
        resolved.append(path)
    return resolved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reject committed imagery and stray tensor files. ADR-0009 §4."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help=(
            "Specific files to check. If omitted, the entire repository "
            "is walked. Non-file paths are an error."
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root (default: derived from script location).",
    )
    args = parser.parse_args(argv)

    root: Path = args.root.resolve()
    if args.paths:
        try:
            targets: Iterable[Path] = _resolve_argv_paths(args.paths)
        except FileNotFoundError as exc:
            print(f"check_no_binaries: {exc}", file=sys.stderr)
            return 2
    else:
        targets = _iter_repo_files(root)

    violations = check_paths(targets, root)
    if violations:
        print("Binary-file CI guard (ADR-0009 §4) failed:", file=sys.stderr)
        for line in violations:
            print(f"  {line}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
