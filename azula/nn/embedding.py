r"""Embedding and encoding modules.

Note:
    The terms embedding and encoding are often used interchangeably in the literature.
    We adopt the following nomenclature: an embedding is a learned function while an
    encoding is a static function.
"""

import torch
import torch.nn as nn

from torch import Tensor

from .utils import promote_dtype


class SineEncoding(nn.Module):
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

    def __init__(self, features: int, omega: float = 1e4):
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
    freqs = omega ** (-freqs)

    return torch.cat(
        (
            torch.sin(x * freqs),
            torch.cos(x * freqs),
        ),
        dim=-1,
    )
