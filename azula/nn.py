r"""Neural networks, layers and modules."""

__all__ = [
    'LayerNorm',
    'SinEmbedding',
    'MLP',
]

import torch
import torch.nn as nn

from torch import Tensor
from typing import Callable, Sequence, Union


class LayerNorm(nn.Module):
    r"""Creates a normalization layer that standardizes features along a dimension.

    .. math:: y = \frac{x - \mathbb{E}[x]}{\sqrt{\mathbb{V}[x] + \epsilon}}

    References:
       Layer Normalization (Lei Ba et al., 2016)
       https://arxiv.org/abs/1607.06450

    Arguments:
        dim: The dimension(s) to standardize.
        eps: A numerical stability term.
    """

    def __init__(self, dim: Union[int, Sequence[int]], eps: float = 1e-5):
        super().__init__()

        self.dim = dim if isinstance(dim, int) else tuple(dim)
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Arguments:
            x: The input tensor :math:`x`, with shape :math:(*).

        Returns:
            The standardized tensor :math:`y`, with shape :math:`(*)`.
        """

        variance, mean = torch.var_mean(x, unbiased=True, dim=self.dim, keepdim=True)

        return (x - mean) / (variance + self.eps).sqrt()


class SinEmbedding(nn.Module):
    r"""Creates a sinusoidal positional embedding.

    .. math::
        e_{2i} & = \sin \left( p \times 10000^\frac{-2i}{D} \right) \\
        e_{2i+1} & = \cos \left( p \times 10000^\frac{-2i}{D} \right)

    References:
        | Attention Is All You Need (Vaswani et al., 2017)
        | https://arxiv.org/abs/1706.03762

    Arguments:
        features: The number of embedding features :math:`D`. Must be even.
    """

    def __init__(self, features: int):
        assert features % 2 == 0

        freqs = torch.linspace(0, 1, features // 2)
        freqs = 1e4 ** (-freqs)

        self.register_buffer('freqs', freqs)

    def forward(self, p: Tensor) -> Tensor:
        r"""
        Arguments:
            p: The position :math:`p`, with shape :math:`(*)`.

        Returns:
            The embedding vector :math:`e`, with shape :math:`(*, D)`.
        """

        p = p[..., None]

        return torch.cat(
            (
                torch.sin(p * self.freqs),
                torch.cos(p * self.freqs),
            ),
            dim=-1,
        )


class MLP(nn.Sequential):
    r"""Creates a multi-layer perceptron (MLP).

    Also known as fully connected feedforward network, an MLP is a sequence of
    non-linear parametric functions

    .. math:: h_{i + 1} = \sigma(W_{i+1} h_i + b_{i+1})

    over feature vectors :math:`h_i`. The non-linear function :math:`\sigma` is called
    an activation function. The trainable parameters of an MLP are its weights
    :math:`W_i \in \mathbb{R}^{D_i \times D_{i-1}}` and biases :math:`b_i \in
    \mathbb{R}^{D_i}`.

    Wikipedia:
        https://wikipedia.org/wiki/Feedforward_neural_network

    Arguments:
        in_features: The number of input features :math:`D_0`.
        out_features: The number of output features :math:`D_L`.
        hid_features: The numbers of hidden features :math:`D_i`.
        activation: The activation function constructor. If :py:`None`, use
            :class:`torch.nn.ReLU` instead.
        normalize: Whether features are normalized between layers or not.
        kwargs: Keyword arguments passed to :class:`torch.nn.Linear`.

    Example:
        >>> net = MLP(64, 1, [32, 16], activation=nn.ELU)
        >>> net
        MLP(
          (0): Linear(in_features=64, out_features=32, bias=True)
          (1): ELU(alpha=1.0)
          (2): Linear(in_features=32, out_features=16, bias=True)
          (3): ELU(alpha=1.0)
          (4): Linear(in_features=16, out_features=1, bias=True)
        )
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hid_features: Sequence[int] = (64, 64),
        activation: Callable[[], nn.Module] = None,
        normalize: bool = False,
        **kwargs,
    ):
        if activation is None:
            activation = nn.ReLU

        if normalize:
            normalization = LayerNorm
        else:
            normalization = lambda: None

        layers = []

        for before, after in zip(
            (in_features, *hid_features),
            (*hid_features, out_features),
        ):
            layers.extend([
                nn.Linear(before, after, **kwargs),
                activation(),
                normalization(),
            ])

        layers = layers[:-2]
        layers = filter(lambda layer: layer is not None, layers)

        super().__init__(*layers)

        self.in_features = in_features
        self.out_features = out_features

    def forward(self, h0: Tensor) -> Tensor:
        r"""
        Arguments:
            h0: The input vector :math:`h_0`, with shape :math:`(*, D_0)`.

        Returns:
            The output vector :math:`h_L`, with shape :math:`(*, D_L)`.
        """

        return super().forward(h0)
