r"""Miscellaneous neural network helpers."""

__all__ = [
    "skip_init",
]

import torch


class skip_init(torch.overrides.TorchFunctionMode):
    r"""Creates a context in which weight initialization is skipped.

    Example:
        >>> with skip_init():
        ...    layer = nn.Linear(3, 5)
    """

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if getattr(func, "__module__", None) == "torch.nn.init":
            if "tensor" in kwargs:
                return kwargs["tensor"]
            else:
                return args[0]
        else:
            return func(*args, **kwargs)
