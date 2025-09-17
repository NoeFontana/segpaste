"""Tests for core functionality."""

from segpaste.core import hello_world


def test_hello_world() -> None:
    """Test the hello_world function."""
    result = hello_world()
    assert isinstance(result, str)
    assert "Hello, World" in result
    assert "segpaste" in result


def test_hello_world_exact_message() -> None:
    """Test the exact message returned by hello_world."""
    expected = "Hello, World from segpaste!"
    assert hello_world() == expected
