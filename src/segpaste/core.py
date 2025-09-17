"""Core functionality for segpaste package."""


def hello_world() -> str:
    """Return a hello world message.

    Returns:
        str: A greeting message.
    """
    return "Hello, World from segpaste!"


def main() -> None:
    """Main entry point for the package."""
    print(hello_world())


if __name__ == "__main__":
    main()
