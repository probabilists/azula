r"""Neural networks, layers and modules."""

__all__ = [
    "LayerNorm",
    "UNetBlock",
    "UNet",
    "SelfAttentionNd",
    "SinEmbedding",
]

import torch
import torch.nn as nn

from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor
from typing import Sequence, Union


class LayerNorm(nn.Module):
    r"""Creates a normalization layer that standardizes features along a dimension.

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

        self.register_buffer("eps", torch.as_tensor(eps))

    def extra_repr(self) -> str:
        return f"dim={self.dim}"

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Arguments:
            x: The input tensor :math:`x`, with shape :math:(*).

        Returns:
            The standardized tensor :math:`y`, with shape :math:`(*)`.
        """

        variance, mean = torch.var_mean(x, dim=self.dim, keepdim=True)

        return (x - mean) / (variance + self.eps).sqrt()


def ConvNd(in_channels: int, out_channels: int, spatial: int = 2, **kwargs) -> nn.Module:
    r"""Creates an N-dimensional convolutional layer.

    Arguments:
        in_channels: Number of input channels :math:`C_i`.
        out_channels: Number of output channels :math:`C_o`.
        spatial: The number of spatial dimensions :math:`N`.
        kwargs: Keyword arguments passed to :class:`torch.nn.Conv2d`.
    """

    CONVS = {
        1: nn.Conv1d,
        2: nn.Conv2d,
        3: nn.Conv3d,
    }

    if spatial in CONVS:
        Conv = CONVS[spatial]
    else:
        raise NotImplementedError()

    return Conv(in_channels, out_channels, **kwargs)


class UNetBlock(nn.Module):
    r"""Creates a residual U-Net block module.

    Arguments:
        channels: The number of channels :math:`C`.
        emb_features: The number of embedding features :math:`D`.
        dropout: The dropout rate.
        spatial: The number of spatial dimensions :math:`N`.
        kwargs: Keyword arguments passed to :class:`ConvNd`.
    """

    def __init__(
        self,
        channels: int,
        emb_features: int,
        dropout: float = None,
        spatial: int = 2,
        **kwargs,
    ):
        super().__init__()

        # Ada-zero
        self.ada_zero = nn.Sequential(
            nn.Linear(emb_features, emb_features),
            nn.SiLU(),
            nn.Linear(emb_features, 3 * channels),
            Rearrange("... (r C) -> r ... C" + " 1" * spatial, r=3),
        )

        layer = self.ada_zero[-2]
        layer.weight = nn.Parameter(layer.weight * 1e-2)

        # Convolutional block
        self.block = nn.Sequential(
            LayerNorm(dim=1),
            ConvNd(channels, channels, spatial=spatial, **kwargs),
            nn.SiLU(),
            nn.Identity() if dropout is None else nn.Dropout(dropout),
            ConvNd(channels, channels, spatial=spatial, **kwargs),
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        r"""
        Arguments:
            x: The input tensor, with shape :math:`(B, C, H_1, ..., H_N)`.
            t: The embedding vector, with shape :math:`(D)` or :math:`(B, D)`.

        Returns:
            The output tensor, with shape :math:`(B, C, H_1, ..., H_N)`.
        """

        a, b, c = self.ada_zero(t)

        y = (a + 1) * x + b
        y = self.block(y)
        y = x + c * y
        y = y / torch.sqrt(1 + c * c)

        return y


class UNet(nn.Module):
    r"""Creates a U-Net module.

    Arguments:
        in_channels: The number of input channels :math:`C_i`.
        out_channels: The number of output channels :math:`C_o`.
        emb_features: The number of embedding features :math:`D`.
        hid_channels: The numbers of channels at each depth.
        hid_blocks: The numbers of hidden blocks at each depth.
        kernel_size: The kernel size for residual blocks.
        dropout: The dropout rate for residual blocks.
        spatial: The number of spatial dimensions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_features: int,
        hid_channels: Sequence[int] = (64, 128, 256),
        hid_blocks: Sequence[int] = (3, 3, 3),
        kernel_size: Union[int, Sequence[int]] = 3,
        dropout: float = None,
        spatial: int = 2,
    ):
        super().__init__()

        assert len(hid_blocks) == len(hid_channels)

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * spatial

        stride = (2,) * spatial
        kwargs = dict(
            kernel_size=tuple(kernel_size),
            padding=tuple(k // 2 for k in kernel_size),
        )

        self.descent, self.ascent = nn.ModuleList(), nn.ModuleList()

        for i, blocks in enumerate(hid_blocks):
            do, up = nn.ModuleList(), nn.ModuleList()

            for _ in range(blocks):
                do.append(
                    UNetBlock(
                        hid_channels[i],
                        emb_features,
                        dropout=dropout,
                        spatial=spatial,
                        **kwargs,
                    )
                )
                up.append(
                    UNetBlock(
                        hid_channels[i],
                        emb_features,
                        dropout=dropout,
                        spatial=spatial,
                        **kwargs,
                    )
                )

            if i > 0:
                do.insert(
                    0,
                    nn.Sequential(
                        ConvNd(
                            hid_channels[i - 1],
                            hid_channels[i],
                            spatial=spatial,
                            stride=stride,
                            **kwargs,
                        ),
                        LayerNorm(dim=1),
                    ),
                )

                up.append(
                    nn.Sequential(
                        LayerNorm(dim=1),
                        nn.Upsample(scale_factor=stride, mode="nearest"),
                    )
                )
            else:
                do.insert(0, ConvNd(in_channels, hid_channels[i], spatial=spatial, **kwargs))
                up.append(ConvNd(hid_channels[i], out_channels, spatial=spatial, kernel_size=1))

            if i + 1 < len(hid_blocks):
                up.insert(
                    0,
                    ConvNd(
                        hid_channels[i] + hid_channels[i + 1],
                        hid_channels[i],
                        **kwargs,
                    ),
                )

            self.descent.append(do)
            self.ascent.insert(0, up)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        r"""
        Arguments:
            x: The input tensor, with shape :math:`(B, C_i, H_1, ..., H_N)`.
            t: The embedding vector, with shape :math:`(D)` or :math:`(B, D)`.

        Returns:
            The output tensor, with shape :math:`(B, C_o, H_1, ..., H_N)`.
        """

        memory = []

        for blocks in self.descent:
            for block in blocks:
                if isinstance(block, UNetBlock):
                    x = block(x, t)
                else:
                    x = block(x)

            memory.append(x)

        for blocks in self.ascent:
            y = memory.pop()
            if x is not y:
                for i in range(2, x.ndim):
                    if x.shape[i] != y.shape[i]:
                        x = torch.narrow(x, i, 0, y.shape[i])

                x = torch.cat((x, y), dim=1)

            for block in blocks:
                if isinstance(block, UNetBlock):
                    x = block(x, t)
                else:
                    x = block(x)

        return x


class SelfAttentionNd(nn.MultiheadAttention):
    r"""Creates an N-dimensional self-attention layer.

    Arguments:
        channels: The number of channels :math:`C`.
        heads: The number of attention heads.
        kwargs: Keyword arguments passed to :class:`torch.nn.MultiheadAttention`.
    """

    def __init__(channels: int, heads: int = 1, **kwargs):
        super().__init__(embed_dim=channels, num_heads=heads, batch_first=True, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Arguments:
            x: The input tensor :math:`x`, with shape :math:`(B, C, H_1, ..., H_N)`.

        Returns:
            The ouput tensor :math:`y`, with shape :math:`(B, C, H_1, ..., H_N)`.
        """

        y = rearrange(x, "B C ...  -> B (...) C")
        y, _ = super().forward(y, y, y, average_attn_weights=False)
        y = rearrange(y, "B L C -> B C L").reshape(x.shape)

        return y


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
        super().__init__()

        assert features % 2 == 0

        freqs = torch.linspace(0, 1, features // 2)
        freqs = 1e4 ** (-freqs)

        self.register_buffer("freqs", freqs)

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
