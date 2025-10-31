r"""Miscellaneous neural network helpers."""

__all__ = [
    "checkpoint",
    "skip_init",
    "cpu_offload",
]

import torch
import torch.nn as nn

from contextlib import contextmanager
from torch._C._functorch import is_gradtrackingtensor
from typing import Callable


class CheckpointReentrant(torch.autograd.Function):
    @staticmethod
    def setup_context(ctx, inputs, outputs):
        func, *xs = inputs

        xs = [x.detach() for x in xs]

        ctx.save_for_backward(*xs)
        ctx.save_for_forward(*xs)
        ctx.func = func

    @staticmethod
    def forward(func, *xs):
        return func(*xs)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def vjp(ctx, *grad_ys):
        xs = ctx.saved_tensors

        with torch.enable_grad():
            xs = [x.detach().requires_grad_() for x in xs]
            ys = ctx.func(*xs)

        if torch.is_tensor(ys):
            ys = [ys]

        grad_xs = torch.autograd.grad(ys, xs, grad_ys)

        return (None, *grad_xs)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def jvp(ctx, grad_func, *grad_xs):
        xs = ctx.saved_tensors

        _, grad_ys = torch.func.jvp(ctx.func, xs, grad_xs)

        return grad_ys


def checkpoint(f: Callable, reentrant: bool = False) -> Callable:
    r"""Applies activation checkpointing to a function.

    Activation checkpointing reduces memory consumption by storing the inputs of the
    function and recomputing its graph during automatic differentiation (AD).

    Reentrant checkpointing is compatible with backward and forward AD, but only
    propagates gradients to the explicit positional inputs of the function. Implicit
    inputs, such as module parameters, do not get gradients. Conversely, non-reentrant
    will propagate gradients to implicit inputs, but is not compatible with foward AD.

    Arguments:
        f: A function.
        reentrant: Whether to use reentrant checkpointing or not.

    Returns:
        The checkpointed function.
    """

    def g(*args, **kwargs):
        mask = [
            torch.is_tensor(arg)
            and torch.is_floating_point(arg)
            and (arg.requires_grad or is_gradtrackingtensor(arg))
            for arg in args
        ]

        tensors = [arg for include, arg in zip(mask, args, strict=True) if include]
        others = [arg for include, arg in zip(mask, args, strict=True) if not include]

        def h(*tensors):
            it, io = iter(tensors), iter(others)
            args = (next(it if include else io) for include in mask)
            return f(*args, **kwargs)

        if reentrant:
            if tensors:
                return CheckpointReentrant.apply(h, *tensors)
            else:
                with torch.no_grad():
                    return h(*tensors)
        else:
            return torch.utils.checkpoint.checkpoint(h, *tensors, use_reentrant=False)

    return g


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
