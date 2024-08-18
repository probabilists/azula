r"""U-Net building blocks."""

__all__ = [
    "UNetBlock",
    "UNet",
]

import torch
import torch.nn as nn

from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor
from typing import Dict, Optional, Sequence, Union

# isort: split
from .normalization import LayerNorm


def ConvNd(in_channels: int, out_channels: int, spatial: int = 2, **kwargs) -> nn.Module:
    r"""Returns an N-dimensional convolutional layer.

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
    r"""Creates a modulated U-Net block module.

    Arguments:
        channels: The number of channels :math:`C`.
        mod_features: The number of modulating features :math:`D`.
        attention_heads: The number of attention heads.
        dropout: The dropout rate in :math:`[0, 1]`.
        spatial: The number of spatial dimensions :math:`N`.
        kwargs: Keyword arguments passed to :class:`torch.nn.Conv2d`.
    """

    def __init__(
        self,
        channels: int,
        mod_features: int,
        attention_heads: Optional[int] = None,
        dropout: Optional[float] = None,
        spatial: int = 2,
        **kwargs,
    ):
        super().__init__()

        # Ada-zero
        self.ada_zero = nn.Sequential(
            nn.Linear(mod_features, mod_features),
            nn.SiLU(),
            nn.Linear(mod_features, 3 * channels),
            Rearrange("... (r C) -> r ... C" + " 1" * spatial, r=3),
        )

        layer = self.ada_zero[-2]
        layer.weight = nn.Parameter(layer.weight * 1e-2)

        # Block
        self.block = nn.Sequential(
            LayerNorm(dim=1),
            ConvNd(channels, channels, spatial=spatial, **kwargs),
            nn.SiLU(),
            nn.Identity() if dropout is None else nn.Dropout(dropout),
            ConvNd(channels, channels, spatial=spatial, **kwargs),
        )

        if attention_heads is not None:
            self.block.extend([
                LayerNorm(dim=1),
                SelfAttentionNd(channels, heads=attention_heads),
            ])

    def forward(self, x: Tensor, mod: Tensor) -> Tensor:
        r"""
        Arguments:
            x: The input tensor, with shape :math:`(B, C, H_1, ..., H_N)`.
            mod: The modulation vector, with shape :math:`(D)` or :math:`(B, D)`.

        Returns:
            The output tensor, with shape :math:`(B, C, H_1, ..., H_N)`.
        """

        a, b, c = self.ada_zero(mod)

        y = (a + 1) * x + b
        y = self.block(y)
        y = x + c * y
        y = y / torch.sqrt(1 + c * c)

        return y


class UNet(nn.Module):
    r"""Creates a modulated U-Net module.

    Arguments:
        in_channels: The number of input channels :math:`C_i`.
        out_channels: The number of output channels :math:`C_o`.
        mod_features: The number of modulating features :math:`D`.
        hid_channels: The numbers of channels at each depth.
        hid_blocks: The numbers of hidden blocks at each depth.
        kernel_size: The kernel size of all convolutions.
        stride: The stride of the downsampling convolutions.
        attention_heads: The number of attention heads at each depth.
        dropout: The dropout rate in :math:`[0, 1]`.
        spatial: The number of spatial dimensions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mod_features: int,
        hid_channels: Sequence[int] = (64, 128, 256),
        hid_blocks: Sequence[int] = (3, 3, 3),
        kernel_size: Union[int, Sequence[int]] = 3,
        stride: Union[int, Sequence[int]] = 2,
        attention_heads: Dict[int, int] = {},  # noqa: B006
        dropout: Optional[float] = None,
        spatial: int = 2,
    ):
        super().__init__()

        assert len(hid_blocks) == len(hid_channels)

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * spatial

        if isinstance(stride, int):
            stride = [stride] * spatial

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
                        mod_features,
                        attention_heads=attention_heads.get(i, None),
                        dropout=dropout,
                        spatial=spatial,
                        **kwargs,
                    )
                )
                up.append(
                    UNetBlock(
                        hid_channels[i],
                        mod_features,
                        attention_heads=attention_heads.get(i, None),
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
                        nn.Upsample(scale_factor=tuple(stride), mode="nearest"),
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

    def forward(self, x: Tensor, mod: Tensor) -> Tensor:
        r"""
        Arguments:
            x: The input tensor, with shape :math:`(B, C_i, H_1, ..., H_N)`.
            mod: The modulation vector, with shape :math:`(D)` or :math:`(B, D)`.

        Returns:
            The output tensor, with shape :math:`(B, C_o, H_1, ..., H_N)`.
        """

        memory = []

        for blocks in self.descent:
            for block in blocks:
                if isinstance(block, UNetBlock):
                    x = block(x, mod)
                else:
                    x = block(x)

            memory.append(x)

        for blocks in self.ascent:
            y = memory.pop()
            if x is not y:
                for i in range(2, x.ndim):
                    if x.shape[i] > y.shape[i]:
                        x = torch.narrow(x, i, 0, y.shape[i])

                x = torch.cat((x, y), dim=1)

            for block in blocks:
                if isinstance(block, UNetBlock):
                    x = block(x, mod)
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

    def __init__(self, channels: int, heads: int = 1, **kwargs):
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
