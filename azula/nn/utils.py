r"""Miscellaneous neural network helpers."""

__all__ = [
    "get_module_dtype",
    "get_module_device",
    "checkpoint",
    "skip_init",
    "promote_dtype",
]

import torch
import torch.nn as nn

from functools import reduce, wraps
from torch._C._functorch import is_gradtrackingtensor
from typing import Callable


def get_module_dtype(module: nn.Module) -> torch.dtype:
    r"""Returns the data type of a module.

    The module's data type is the first floating-point type in the module's parameters
    or buffers. If there is none, returns :py:`None`.

    Arguments:
        module: A module.
    """

    for p in module.parameters():
        if torch.is_floating_point(p):
            return p.dtype

    for b in module.buffers():
        if torch.is_floating_point(b):
            return b.dtype

    return None


def get_module_device(module: nn.Module) -> torch.device:
    r"""Returns the execution device of a module.

    The module's device is the first device in the module's parameters or buffers. If
    there is none, returns :py:`None`.

    Arguments:
        module: A module.
    """

    for m in module.modules():
        if hasattr(m, "_hf_hook") and hasattr(m._hf_hook, "execution_device"):
            if m._hf_hook.execution_device is not None:
                return m._hf_hook.execution_device

        for p in m.parameters(recurse=False):
            if not p.is_meta():
                return p.device

        for b in m.parameters(recurse=False):
            if not b.is_meta():
                return b.device

    return None


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

    @wraps(f)
    def g(*args, **kwargs):
        mask = [
            torch.is_tensor(arg)
            and torch.is_floating_point(arg)
            and (arg.requires_grad or is_gradtrackingtensor(arg))
            for arg in args
        ]

        tensors = [arg for include, arg in zip(mask, args) if include]
        others = [arg for include, arg in zip(mask, args) if not include]

        def h(*tensors):
            it, io = iter(tensors), iter(others)
            args = (next(it if include else io) for include in mask)
            return f(*args, **kwargs)

        if any(map(is_gradtrackingtensor, tensors)):
            return h(*tensors)
        elif reentrant:
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


def promote_dtype(f: Callable, min_dtype: torch.dtype = torch.float32) -> Callable:
    r"""Applies data type promotion to a function.

    Arguments:
        f: A function.
        min_dtype: The minimum precision data type.

    Returns:
        The promoted function.
    """

    @wraps(f)
    def g(*args, **kwargs):
        dtypes = [arg.dtype for arg in args]
        dtype = reduce(torch.promote_types, dtypes)

        args = [arg.to(torch.promote_types(arg.dtype, min_dtype)) for arg in args]
        outs = f(*args, **kwargs)

        if torch.is_tensor(outs):
            return outs.to(dtype)
        else:
            return tuple(out.to(dtype) for out in outs)

    return g
