r"""Common layers."""

__all__ = [
    "ConvNd",
    "ReLU2",
    "LayerNorm",
    "RMSNorm",
    "Patchify",
    "Unpatchify",
]

import string
import torch
import torch.nn as nn

from einops.layers.torch import Rearrange
from torch import Tensor
from typing import Sequence, Union

from .utils import promote_dtype


def ConvNd(
    in_channels: int,
    out_channels: int,
    spatial: int = 2,
    identity_init: bool = False,
    **kwargs,
) -> nn.Module:
    r"""Returns an N-dimensional convolutional layer.

    Arguments:
        in_channels: The number of input channels :math:`C_i`.
        out_channels: The number of output channels :math:`C_o`.
        spatial: The number of spatial dimensions :math:`N`.
        identity_init: Initialize the convolution as a (pseudo-)identity.
        kwargs: Keyword arguments passed to :class:`torch.nn.Conv2d`.
    """

    CONVS = {
        0: nn.Linear,
        1: nn.Conv1d,
        2: nn.Conv2d,
        3: nn.Conv3d,
    }

    if spatial in CONVS:
        Conv = CONVS[spatial]
    else:
        raise NotImplementedError()

    conv = Conv(in_channels, out_channels, **kwargs)

    if identity_init:
        kernel_size = conv.weight.shape[2:]
        kernel_center = [k // 2 for k in kernel_size]

        eye = torch.zeros_like(conv.weight.data[:in_channels])

        for i in range(min(in_channels, out_channels)):
            eye[tuple((i, i, *kernel_center))] = 1.0

        conv.weight.data[:in_channels].mul_(1e-2)
        conv.weight.data[:in_channels].add_(eye)

    return conv


class ReLU2(nn.Module):
    r"""Creates a ReLU² activation layer.

    .. math:: y = \max(x, 0)^2

    References:
        | Primer: Searching for Efficient Transformers for Language Modeling (So et al., 2021)
        | https://arxiv.org/abs/2109.08668
    """

    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.relu(x).square()


class LayerNorm(nn.Module):
    r"""Creates a layer that standardizes features along a dimension.

    .. math:: y = \frac{x - \mathbb{E}[x]}{\sqrt{\mathbb{V}[x] + \epsilon}}

    References:
       | Layer Normalization (Lei Ba et al., 2016)
       | https://arxiv.org/abs/1607.06450

    Arguments:
        dim: The dimension(s) to standardize.
        eps: A numerical stability term.
    """

    def __init__(self, dim: Union[int, Sequence[int]], eps: float = 1e-5):
        super().__init__()

        self.dim = dim if isinstance(dim, int) else tuple(dim)
        self.eps = eps

    def extra_repr(self) -> str:
        return f"dim={self.dim}"

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Arguments:
            x: The input tensor :math:`x`, with shape :math:(*).

        Returns:
            The standardized tensor :math:`y`, with shape :math:`(*)`.
        """

        return layer_norm(x, dim=self.dim, eps=self.eps)


@promote_dtype
def layer_norm(x: Tensor, /, dim: Sequence[int], eps: float = 1e-5) -> Tensor:
    v, m = torch.var_mean(x, dim=dim, keepdim=True)
    return (x - m) * torch.rsqrt(v + eps)


class RMSNorm(nn.Module):
    r"""Creates a layer that normalizes features along a dimension.

    .. math:: y = \frac{x}{\sqrt{\mathbb{E}[x^2] + \epsilon}}

    References:
       | Root Mean Square Layer Normalization (Zhang et al., 2019)
       | https://arxiv.org/abs/1910.07467

    Arguments:
        dim: The dimension(s) to normalize.
        eps: A numerical stability term.
    """

    def __init__(self, dim: Union[int, Sequence[int]], eps: float = 1e-5):
        super().__init__()

        self.dim = dim if isinstance(dim, int) else tuple(dim)
        self.eps = eps

    def extra_repr(self) -> str:
        return f"dim={self.dim}"

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Arguments:
            x: The input tensor :math:`x`, with shape :math:(*).

        Returns:
            The normalized tensor :math:`y`, with shape :math:`(*)`.
        """

        return rms_norm(x, dim=self.dim, eps=self.eps)


@promote_dtype
def rms_norm(x: Tensor, /, dim: Sequence[int], eps: float = 1e-5) -> Tensor:
    return x * torch.rsqrt(torch.mean(torch.square(x), dim=dim, keepdim=True) + eps)


def Patchify(patch_shape: Sequence[int], channel_last: bool = False) -> Rearrange:
    r"""Returns a patch-to-channel layer.

    Arguments:
        patch_shape: The patch shape.
        channel_last: Whether the output channel dimension is first or last.
    """

    ndim = len(patch_shape)

    ABC = string.ascii_uppercase[:ndim]
    abc = string.ascii_lowercase[:ndim]

    in_shape = (f"({A} {a})" for A, a in zip(ABC, abc))
    in_shape = "... Z " + " ".join(in_shape)

    if channel_last:
        out_shape = "... " + " ".join(ABC) + " (Z " + " ".join(abc) + ")"
    else:
        out_shape = "... (Z " + " ".join(abc) + ") " + " ".join(ABC)

    lengths = {a: size for a, size in zip(abc, patch_shape)}

    return Rearrange(f"{in_shape} -> {out_shape}", **lengths)


def Unpatchify(patch_shape: Sequence[int], channel_last: bool = False) -> Rearrange:
    r"""Returns a channel-to-patch layer.

    Arguments:
        patch_shape: The patch shape.
        channel_last: Whether the input channel dimension is first or last.
    """

    ndim = len(patch_shape)

    ABC = string.ascii_uppercase[:ndim]
    abc = string.ascii_lowercase[:ndim]

    in_shape = (f"({A} {a})" for A, a in zip(ABC, abc))
    in_shape = "... Z " + " ".join(in_shape)

    if channel_last:
        out_shape = "... " + " ".join(ABC) + " (Z " + " ".join(abc) + ")"
    else:
        out_shape = "... (Z " + " ".join(abc) + ") " + " ".join(ABC)

    lengths = {a: size for a, size in zip(abc, patch_shape)}

    return Rearrange(f"{out_shape} -> {in_shape}", **lengths)
