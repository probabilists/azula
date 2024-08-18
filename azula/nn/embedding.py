r"""Embedding and encoding modules.

Note:
    The terms embedding and encoding are often used interchangeably in the literature.
    We adopt the following nomenclature: an embedding is a learned function while an
    encoding is a static function.
"""

import torch
import torch.nn as nn

from torch import Tensor


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

        freqs = torch.linspace(0, 1, features // 2)
        freqs = omega ** (-freqs)

        self.register_buffer("freqs", freqs)

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Arguments:
            x: The position :math:`x`, with shape :math:`(*)`.

        Returns:
            The embedding vector :math:`e`, with shape :math:`(*, D)`.
        """

        x = x.unsqueeze(dim=-1)

        return torch.cat(
            (
                torch.sin(x * self.freqs),
                torch.cos(x * self.freqs),
            ),
            dim=-1,
        )
