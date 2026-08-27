r"""U-Net building blocks."""

__all__ = [
    "UNet",
    "UNetBlock",
]

import torch

from collections.abc import Sequence
from einops.layers.torch import Rearrange
from torch import Tensor

from .layers import ConvNd, RMSNorm
from .utils import checkpoint


class UNetBlock(torch.nn.Module):
    r"""Creates a modulated U-Net block module.

    Arguments:
        channels: The number of channels :math:`C`.
        mod_features: The number of modulating features :math:`D`.
        ffn_factor: The channel factor in the FFN.
        spatial: The number of spatial dimensions :math:`N`.
        dropout: The dropout rate in :math:`[0, 1]`.
        checkpointing: Whether to use activation checkpointing or not.
        kwargs: Keyword arguments passed to :class:`azula.nn.layers.ConvNd`.
    """

    def __init__(
        self,
        channels: int,
        mod_features: int = 0,
        ffn_factor: int = 1,
        spatial: int = 2,
        dropout: float | None = None,
        checkpointing: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.checkpointing = checkpointing

        # Ada-Norm Zero
        self.norm = RMSNorm(dim=-spatial - 1)

        if mod_features > 0:
            self.ada_zero = torch.nn.Sequential(
                torch.nn.Linear(mod_features, 3 * channels),
                Rearrange("... (n C) -> n ... C" + " 1" * spatial, n=3),
            )
            self.ada_zero[0].weight.data.mul_(1e-2)
            self.ada_zero[0].bias.data.zero_()
        else:
            self.ada_zero = None

        # Block
        self.ffn = torch.nn.Sequential(
            ConvNd(channels, ffn_factor * channels, spatial=spatial, **kwargs),
            torch.nn.SiLU(),
            torch.nn.Identity() if dropout is None else torch.nn.Dropout(dropout),
            ConvNd(ffn_factor * channels, channels, spatial=spatial, **kwargs),
        )

        if self.ada_zero is None:
            self.ffn[-1].weight.data.mul_(1e-2)
            self.ffn[-1].bias.data.zero_()

    def _forward(self, x: Tensor, mod: Tensor | None = None) -> Tensor:
        if self.ada_zero is None:
            y = x + self.ffn(self.norm(x))
        else:
            a, b, c = self.ada_zero(mod)

            y = (a + 1) * self.norm(x) + b
            y = self.ffn(y)
            y = x + c * y

        return y

    def forward(
        self,
        x: Tensor,
        mod: Tensor | None = None,
    ) -> Tensor:
        r"""
        Arguments:
            x: The input tensor, with shape :math:`(B, C, L_1, ..., L_N)`.
            mod: The modulation vector, with shape :math:`(D)` or :math:`(B, D)`.

        Returns:
            The output tensor, with shape :math:`(B, C, L_1, ..., L_N)`.
        """

        if self.checkpointing:
            return checkpoint(self._forward, reentrant=not self.training)(x, mod)
        else:
            return self._forward(x, mod)


class UNet(torch.nn.Module):
    r"""Creates a modulated U-Net module.

    Arguments:
        in_channels: The number of input channels :math:`C_i`.
        out_channels: The number of output channels :math:`C_o`.
        cond_channels: The number of condition channels :math:`C_c`.
        mod_features: The number of modulating features :math:`D`.
        hid_channels: The numbers of channels at each depth.
        hid_blocks: The numbers of hidden blocks at each depth.
        kernel_size: The kernel size of all convolutions.
        stride: The stride of the downsampling convolutions.
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
        mod_features: int = 0,
        hid_channels: Sequence[int] = (64, 128, 256),
        hid_blocks: Sequence[int] = (3, 3, 3),
        kernel_size: int | Sequence[int] = 3,
        stride: int | Sequence[int] = 2,
        spatial: int = 2,
        periodic: bool = False,
        identity_init: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        assert len(hid_blocks) == len(hid_channels)

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * spatial

        if isinstance(stride, int):
            stride = [stride] * spatial

        kwargs = {**kwargs, "mod_features": mod_features}
        conv_kwargs = {
            "kernel_size": tuple(kernel_size),
            "padding": tuple(k // 2 for k in kernel_size),
            "padding_mode": "circular" if periodic else "zeros",
            "spatial": spatial,
        }

        self.in_proj = ConvNd(in_channels + cond_channels, hid_channels[0], **conv_kwargs)
        self.out_proj = ConvNd(hid_channels[0], out_channels, **conv_kwargs)
        self.out_norm = RMSNorm(dim=-spatial - 1)

        if mod_features > 0:
            self.out_ada_zero = torch.nn.Sequential(
                torch.nn.Linear(mod_features, 2 * hid_channels[0]),
                Rearrange("... (n C) -> n ... C" + " 1" * spatial, n=2),
            )
            self.out_ada_zero[0].weight.data.mul_(1e-2)
            self.out_ada_zero[0].bias.data.zero_()
        else:
            self.out_ada_zero = None

        self.descent, self.ascent = torch.nn.ModuleList(), torch.nn.ModuleList()

        for i, num_blocks in enumerate(hid_blocks):
            do, up = torch.nn.ModuleList(), torch.nn.ModuleList()

            if i > 0:
                do.append(RMSNorm(dim=-spatial - 1))
                do.append(
                    ConvNd(
                        hid_channels[i - 1],
                        hid_channels[i],
                        stride=stride,
                        identity_init=identity_init,
                        **conv_kwargs,
                    )
                )

            if i + 1 < len(hid_blocks):
                up.append(
                    ConvNd(
                        hid_channels[i] + hid_channels[i + 1],
                        hid_channels[i],
                        identity_init=identity_init,
                        **conv_kwargs,
                    )
                )

            for _ in range(num_blocks):
                do.append(UNetBlock(hid_channels[i], **conv_kwargs, **kwargs))
                up.append(UNetBlock(hid_channels[i], **conv_kwargs, **kwargs))

            if i > 0:
                up.append(RMSNorm(dim=-spatial - 1))
                up.append(torch.nn.Upsample(scale_factor=tuple(stride), mode="nearest"))

            self.descent.append(do)
            self.ascent.insert(0, up)

    def forward(
        self,
        x: Tensor,
        mod: Tensor | None = None,
        cond: Tensor | None = None,
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

        x = self.in_proj(x)

        memory = []

        for blocks in self.descent:
            for block in blocks:
                if isinstance(block, UNetBlock):
                    x = block(x, mod)
                else:
                    x = block(x)

            memory.append(x)

        memory[-1] = None

        if hasattr(self, "bottleneck"):
            x = self.bottleneck(x, mod)

        for blocks in self.ascent:
            y = memory.pop()

            if y is not None:
                for i in range(2, x.ndim):
                    if x.shape[i] > y.shape[i]:
                        x = torch.narrow(x, i, 0, y.shape[i])

                x = torch.cat((y, x), dim=1)

            for block in blocks:
                if isinstance(block, UNetBlock):
                    x = block(x, mod)
                else:
                    x = block(x)

        if self.out_ada_zero is None:
            x = self.out_norm(x)
        else:
            a, b = self.out_ada_zero(mod)
            x = (a + 1) * self.out_norm(x) + b

        x = self.out_proj(x)

        return x
