from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

import torch.compiler

F = TypeVar("F", bound=Callable[..., Any])


def skip_if_compiling(func: F) -> F:
    """
    A decorator that skips the execution of the decorated function when
    torch.compile is active.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not torch.compiler.is_compiling():
            return func(*args, **kwargs)
        # When compiling, do nothing.
        return None

    return wrapper  # type: ignore[return-value]
