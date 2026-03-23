r"""Common layers."""

__all__ = [
    "ConvNd",
    "LayerNorm",
    "Patchify",
    "RMSNorm",
    "ReLU2",
    "SineEncoding",
    "SwiGLU",
    "Unpatchify",
]

import math
import string
import torch

from collections.abc import Sequence
from einops.layers.torch import Rearrange
from torch import Tensor

from .utils import promote_dtype


def ConvNd(
    in_channels: int,
    out_channels: int,
    spatial: int = 2,
    identity_init: bool = False,
    **kwargs,
) -> torch.nn.Module:
    r"""Returns an N-dimensional convolutional layer.

    Arguments:
        in_channels: The number of input channels :math:`C_i`.
        out_channels: The number of output channels :math:`C_o`.
        spatial: The number of spatial dimensions :math:`N`.
        identity_init: Initialize the convolution as a (pseudo-)identity.
        kwargs: Keyword arguments passed to :class:`torch.nn.Conv2d`.
    """

    CONVS = {
        0: torch.nn.Linear,
        1: torch.nn.Conv1d,
        2: torch.nn.Conv2d,
        3: torch.nn.Conv3d,
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
            eye[(i, i, *kernel_center)] = 1.0

        conv.weight.data[:in_channels].mul_(1e-2)
        conv.weight.data[:in_channels].add_(eye)

    return conv


class ReLU2(torch.nn.Module):
    r"""Creates a ReLU² activation layer.

    .. math:: y = \max(x, 0)^2

    References:
        | Primer: Searching for Efficient Transformers for Language Modeling (So et al., 2021)
        | https://arxiv.org/abs/2109.08668
    """

    def forward(self, x: Tensor) -> Tensor:
        return relu2(x)


def relu2(x: Tensor, /) -> Tensor:
    return torch.nn.functional.relu(x).square()


class SwiGLU(torch.nn.Module):
    r"""Creates a SwiGLU activation layer.

    .. math:: y = x_1 \times x_2 \times \sigma(x_2)

    References:
        | GLU Variants Improve Transformer (Shazeer, 2020)
        | https://arxiv.org/abs/2002.05202
    """

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Arguments:
            x: The input tensor :math:`x`, with shape :math:`(*, 2C)`.

        Returns:
            The output tensor :math:`y`, with shape :math:`(*, C)`.
        """

        return swiglu(x)


def swiglu(x: Tensor, /) -> Tensor:
    x = x.unflatten(-1, (-1, 2))
    x1, x2 = x[..., 0], x[..., 1]
    return x1 * torch.nn.functional.silu(x2)


class LayerNorm(torch.nn.Module):
    r"""Creates a layer that standardizes features along a dimension.

    .. math:: y = \frac{x - \mathbb{E}[x]}{\sqrt{\mathbb{V}[x] + \epsilon}}

    References:
       | Layer Normalization (Lei Ba et al., 2016)
       | https://arxiv.org/abs/1607.06450

    Arguments:
        dim: The dimension(s) to standardize.
        eps: A numerical stability term.
    """

    def __init__(self, dim: int | Sequence[int], eps: float = 1e-5) -> None:
        super().__init__()

        self.dim = dim
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
def layer_norm(x: Tensor, /, dim: int = -1, eps: float = 1e-5) -> Tensor:
    v, m = torch.var_mean(x, dim=dim, keepdim=True)
    return (x - m) * torch.rsqrt(v + eps)


class RMSNorm(torch.nn.Module):
    r"""Creates a layer that normalizes features along a dimension.

    .. math:: y = \frac{x}{\sqrt{\mathbb{E}[x^2] + \epsilon}}

    References:
       | Root Mean Square Layer Normalization (Zhang et al., 2019)
       | https://arxiv.org/abs/1910.07467

    Arguments:
        dim: The dimension(s) to normalize.
        eps: A numerical stability term.
    """

    def __init__(self, dim: int | Sequence[int], eps: float = 1e-5) -> None:
        super().__init__()

        self.dim = dim
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
def rms_norm(x: Tensor, /, dim: int = -1, eps: float = 1e-5) -> Tensor:
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

    in_shape = (f"({A} {a})" for A, a in zip(ABC, abc, strict=True))
    in_shape = "... Z " + " ".join(in_shape)

    if channel_last:
        out_shape = "... " + " ".join(ABC) + " (Z " + " ".join(abc) + ")"
    else:
        out_shape = "... (Z " + " ".join(abc) + ") " + " ".join(ABC)

    lengths = {a: size for a, size in zip(abc, patch_shape, strict=True)}

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

    in_shape = (f"({A} {a})" for A, a in zip(ABC, abc, strict=True))
    in_shape = "... Z " + " ".join(in_shape)

    if channel_last:
        out_shape = "... " + " ".join(ABC) + " (Z " + " ".join(abc) + ")"
    else:
        out_shape = "... (Z " + " ".join(abc) + ") " + " ".join(ABC)

    lengths = {a: size for a, size in zip(abc, patch_shape, strict=True)}

    return Rearrange(f"{out_shape} -> {in_shape}", **lengths)


class SineEncoding(torch.nn.Module):
    r"""Creates a sinusoidal positional encoding.

    .. math::
        e_{2i} & = \sin \left( x \times \omega^\frac{-2i}{D} \right) \\
        e_{2i+1} & = \cos \left( x \times \omega^\frac{-2i}{D} \right)

    References:
        | Attention Is All You Need (Vaswani et al., 2017)
        | https://arxiv.org/abs/1706.03762

    Arguments:
        features: The number of embedding features :math:`D`. Must be even.
        omega: The maximum frequency :math:`\omega`.
    """

    def __init__(self, features: int, omega: float = 1e4) -> None:
        super().__init__()

        assert features % 2 == 0

        self.features = features
        self.omega = omega

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Arguments:
            x: The position :math:`x`, with shape :math:`(*)`.

        Returns:
            The embedding vector :math:`e`, with shape :math:`(*, D)`.
        """

        return sine_encoding(x, features=self.features, omega=self.omega)


@promote_dtype
def sine_encoding(x: Tensor, /, features: int, omega: float = 1e4) -> Tensor:
    x = x.unsqueeze(dim=-1)

    freqs = torch.linspace(0, 1, features // 2, dtype=x.dtype, device=x.device)
    freqs = torch.exp(math.log(1 / omega) * freqs)

    return torch.cat(
        (
            torch.sin(x * freqs),
            torch.cos(x * freqs),
        ),
        dim=-1,
    )
