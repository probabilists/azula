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
from torch.utils.checkpoint import checkpoint
from typing import Dict, Optional, Sequence, Union

from .attention import MultiheadSelfAttention
from .layers import ConvNd, LayerNorm


class UNetBlock(nn.Module):
    r"""Creates a modulated U-Net block module.

    Arguments:
        channels: The number of channels :math:`C`.
        mod_features: The number of modulating features :math:`D`.
        norm: The kind of normalization.
        groups: The number of groups in :class:`torch.nn.GroupNorm` layers.
        attention_heads: The number of attention heads.
        ffn_factor: The channel factor in the FFN.
        spatial: The number of spatial dimensions :math:`N`.
        dropout: The dropout rate in :math:`[0, 1]`.
        checkpointing: Whether to use gradient checkpointing or not.
        kwargs: Keyword arguments passed to :class:`azula.nn.layers.ConvNd`.
    """

    def __init__(
        self,
        channels: int,
        mod_features: int = 0,
        norm: str = "layer",
        groups: int = 16,
        attention_heads: Optional[int] = None,
        ffn_factor: int = 1,
        spatial: int = 2,
        dropout: Optional[float] = None,
        checkpointing: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.checkpointing = checkpointing

        # Attention
        if attention_heads is None:
            self.attn = None
        else:
            self.attn = SelfAttentionNd(channels, attention_heads=attention_heads)

        # Ada-Norm Zero
        if norm == "layer":
            self.norm = LayerNorm(dim=-spatial - 1)
        elif norm == "group":
            self.norm = nn.GroupNorm(
                num_groups=min(groups, channels),
                num_channels=channels,
                affine=False,
            )
        else:
            raise NotImplementedError()

        if mod_features > 0:
            self.ada_zero = nn.Sequential(
                nn.Linear(mod_features, mod_features),
                nn.SiLU(),
                nn.Linear(mod_features, 3 * channels),
                Rearrange("... (n C) -> n ... C" + " 1" * spatial, n=3),
            )

            self.ada_zero[-2].weight.data.mul_(1e-2)
        else:
            self.ada_zero = nn.Parameter(torch.randn(3, channels, *(1,) * spatial))
            self.ada_zero.data.mul_(1e-2)

        # Block
        self.ffn = nn.Sequential(
            ConvNd(channels, ffn_factor * channels, spatial=spatial, **kwargs),
            nn.SiLU(),
            nn.Identity() if dropout is None else nn.Dropout(dropout),
            ConvNd(ffn_factor * channels, channels, spatial=spatial, **kwargs),
        )

    def _forward(self, x: Tensor, mod: Optional[Tensor] = None) -> Tensor:
        if torch.is_tensor(self.ada_zero):
            a, b, c = self.ada_zero
        else:
            a, b, c = self.ada_zero(mod)

        y = (a + 1) * self.norm(x) + b
        y = y if self.attn is None else y + self.attn(y)
        y = self.ffn(y)
        y = (x + c * y) * torch.rsqrt(1 + c * c)

        return y

    def forward(
        self,
        x: Tensor,
        mod: Optional[Tensor] = None,
    ) -> Tensor:
        r"""
        Arguments:
            x: The input tensor, with shape :math:`(B, C, L_1, ..., L_N)`.
            mod: The modulation vector, with shape :math:`(D)` or :math:`(B, D)`.

        Returns:
            The output tensor, with shape :math:`(B, C, L_1, ..., L_N)`.
        """

        if self.checkpointing:
            return checkpoint(self._forward, x, mod, use_reentrant=False)
        else:
            return self._forward(x, mod)


class UNet(nn.Module):
    r"""Creates a modulated U-Net module.

    Arguments:
        in_channels: The number of input channels :math:`C_i`.
        out_channels: The number of output channels :math:`C_o`.
        cond_channels: The number of condition channels :math:`C_c`.
        hid_channels: The numbers of channels at each depth.
        hid_blocks: The numbers of hidden blocks at each depth.
        kernel_size: The kernel size of all convolutions.
        stride: The stride of the downsampling convolutions.
        attention_heads: The number of attention heads at each depth.
        spatial: The number of spatial dimensions :math:`N`.
        periodic: Whether the spatial dimensions are periodic or not.
        identity_init: Initialize down/upsampling convolutions as identity.
        kwargs: Keyword arguments passed to :class:`UNetBlock`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_channels: int = 0,
        hid_channels: Sequence[int] = (64, 128, 256),
        hid_blocks: Sequence[int] = (3, 3, 3),
        kernel_size: Union[int, Sequence[int]] = 3,
        stride: Union[int, Sequence[int]] = 2,
        attention_heads: Dict[int, int] = {},  # noqa: B006
        spatial: int = 2,
        periodic: bool = False,
        identity_init: bool = True,
        **kwargs,
    ):
        super().__init__()

        assert len(hid_blocks) == len(hid_channels)

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * spatial

        if isinstance(stride, int):
            stride = [stride] * spatial

        conv_kwargs = dict(
            kernel_size=tuple(kernel_size),
            padding=tuple(k // 2 for k in kernel_size),
            padding_mode="circular" if periodic else "zeros",
            spatial=spatial,
        )

        self.descent, self.ascent = nn.ModuleList(), nn.ModuleList()

        for i, num_blocks in enumerate(hid_blocks):
            do, up = nn.ModuleList(), nn.ModuleList()

            for _ in range(num_blocks):
                do.append(
                    UNetBlock(
                        hid_channels[i],
                        attention_heads=attention_heads.get(i, None),
                        **conv_kwargs,
                        **kwargs,
                    )
                )

                up.append(
                    UNetBlock(
                        hid_channels[i],
                        attention_heads=attention_heads.get(i, None),
                        **conv_kwargs,
                        **kwargs,
                    )
                )

            if i > 0:
                do.insert(
                    0,
                    ConvNd(
                        hid_channels[i - 1],
                        hid_channels[i],
                        stride=stride,
                        identity_init=identity_init,
                        **conv_kwargs,
                    ),
                )

                up.append(nn.Upsample(scale_factor=tuple(stride), mode="nearest"))
            else:
                do.insert(0, ConvNd(in_channels + cond_channels, hid_channels[i], **conv_kwargs))
                up.append(ConvNd(hid_channels[i], out_channels, **conv_kwargs))

            if i + 1 < len(hid_blocks):
                up.insert(
                    0,
                    ConvNd(
                        hid_channels[i] + hid_channels[i + 1],
                        hid_channels[i],
                        identity_init=identity_init,
                        **conv_kwargs,
                    ),
                )

            self.descent.append(do)
            self.ascent.insert(0, up)

    def forward(
        self,
        x: Tensor,
        mod: Optional[Tensor] = None,
        cond: Optional[Tensor] = None,
    ) -> Tensor:
        r"""
        Arguments:
            x: The input tensor, with shape :math:`(B, C_i, L_1, ..., L_N)`.
            mod: The modulation vector, with shape :math:`(D)` or :math:`(B, D)`.
            cond: The condition tensor, with shape :math:`(B, C_c, L_1, ..., L_N)`.

        Returns:
            The output tensor, with shape :math:`(B, C_o, L_1, ..., L_N)`.
        """

        if cond is not None:
            x = torch.cat((x, cond), dim=1)

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


class SelfAttentionNd(MultiheadSelfAttention):
    r"""Creates an N-dimensional self-attention layer."""

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Arguments:
            x: The input tensor :math:`x`, with shape :math:`(B, C, L_1, ..., L_N)`.

        Returns:
            The ouput tensor :math:`y`, with shape :math:`(B, C, L_1, ..., L_N)`.
        """

        y = rearrange(x, "B C ...  -> B (...) C")
        y = super().forward(y)
        y = rearrange(y, "B L C -> B C L").reshape(x.shape)

        return y
