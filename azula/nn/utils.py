r"""Miscellaneous neural network helpers."""

__all__ = [
    "skip_init",
    "cpu_offload",
]

import torch
import torch.nn as nn

from contextlib import contextmanager


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


@contextmanager
def cpu_offload(module: nn.Module, device: torch.device):
    r"""Moves a module to a device and offloads it to CPU upon exit.

    Arguments:
        module: The module to offload.
        device: The target device.
    """

    try:
        yield module.to(device=device)
    finally:
        module.cpu()
