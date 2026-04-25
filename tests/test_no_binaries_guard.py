"""Behavior of `scripts/check_no_binaries.py` (ADR-0009 §4)."""

from __future__ import annotations

import contextlib
import importlib.util
import io
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "check_no_binaries.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_no_binaries", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_GUARD = _load_script()


def _run(*paths: Path, root: Path | None = None) -> tuple[int, str]:
    """Invoke the guard in-process; return (exit_code, captured stderr)."""
    argv: list[str] = []
    if root is not None:
        argv.extend(["--root", str(root)])
    argv.extend(str(p) for p in paths)
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        code = _GUARD.main(argv)
    return code, buf.getvalue()


@pytest.mark.parametrize(
    "filename",
    [
        "snapshot.png",
        "render.JPG",
        "thumb.jpeg",
        "scene.tiff",
        "scene.tif",
        "image.webp",
        "icon.bmp",
        "frame.gif",
    ],
)
def test_image_extensions_are_rejected(tmp_path: Path, filename: str) -> None:
    bad = tmp_path / filename
    bad.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    code, stderr = _run(bad, root=tmp_path)
    assert code == 1
    assert "image files are forbidden" in stderr
    assert filename in stderr


def test_tensor_outside_allowlist_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "model.pt"
    bad.write_bytes(b"\x80\x02}q\x00.")
    code, stderr = _run(bad, root=tmp_path)
    assert code == 1
    assert "tensor / checkpoint files allowed only under" in stderr


def test_tensor_inside_allowlist_is_accepted(tmp_path: Path) -> None:
    fixtures = tmp_path / "tests" / "fixtures" / "synthetic"
    fixtures.mkdir(parents=True)
    ok = fixtures / "ks_snapshot.pt"
    ok.write_bytes(b"\x80\x02}q\x00.")
    code, stderr = _run(ok, root=tmp_path)
    assert code == 0, stderr


def test_tensor_at_fixtures_root_is_accepted(tmp_path: Path) -> None:
    """``tests/fixtures/ks_snapshot.pt`` (ADR-0008 §D6) is in-tree."""
    fixtures = tmp_path / "tests" / "fixtures"
    fixtures.mkdir(parents=True)
    ok = fixtures / "ks_snapshot.pt"
    ok.write_bytes(b"\x80\x02}q\x00.")
    code, stderr = _run(ok, root=tmp_path)
    assert code == 0, stderr


def test_text_files_are_accepted(tmp_path: Path) -> None:
    ok = tmp_path / "notes.md"
    ok.write_text("# notes\n")
    code, stderr = _run(ok, root=tmp_path)
    assert code == 0, stderr


def test_repo_walk_passes_on_current_tree() -> None:
    """The committed tree must already satisfy the guard."""
    code, stderr = _run()
    assert code == 0, "binary-file guard tripped on the committed tree:\n" + stderr


def test_violations_list_every_offender(tmp_path: Path) -> None:
    a = tmp_path / "a.png"
    b = tmp_path / "b.jpg"
    a.write_bytes(b"\x00")
    b.write_bytes(b"\x00")
    code, stderr = _run(a, b, root=tmp_path)
    assert code == 1
    assert "a.png" in stderr
    assert "b.jpg" in stderr


def test_missing_argv_path_errors(tmp_path: Path) -> None:
    """Argv-supplied paths that don't exist now error instead of silently passing."""
    code, stderr = _run(tmp_path / "does-not-exist.png", root=tmp_path)
    assert code == 2
    assert "not a file" in stderr


def test_directory_argv_is_rejected_as_non_file(tmp_path: Path) -> None:
    sub = tmp_path / "subdir"
    sub.mkdir()
    code, stderr = _run(sub, root=tmp_path)
    assert code == 2
    assert "not a file" in stderr


def test_walk_prunes_excluded_directories(tmp_path: Path) -> None:
    """A PNG buried inside ``.venv`` / ``__pycache__`` does not trip the guard."""
    venv = tmp_path / ".venv" / "site-packages"
    venv.mkdir(parents=True)
    (venv / "wheel-icon.png").write_bytes(b"\x00")
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "compiled.pyc").write_bytes(b"\x00")
    code, stderr = _run(root=tmp_path)
    assert code == 0, stderr
