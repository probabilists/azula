r"""Diffusion Transformer (DiT) building blocks.

References:
    | Scalable Diffusion Models with Transformers (Peebles et al., 2022)
    | https://arxiv.org/abs/2212.09748
"""

__all__ = [
    "DiT",
    "DiTBlock",
]

import torch

from einops.layers.torch import Rearrange
from torch import Tensor
from typing import Literal

from .attention import MultiheadSelfAttention
from .layers import ReLU2, RMSNorm, SineEncoding, SwiGLU
from .utils import checkpoint


class DiTBlock(torch.nn.Module):
    r"""Creates a modulated DiT block module.

    Arguments:
        channels: The number of channels :math:`C`.
        mod_features: The number of modulating features :math:`D`.
        ffn_factor: The channel factor in the FFN.
        ffn_activation: The activation function in the FFN.
        dropout: The dropout rate in :math:`[0, 1]`.
        checkpointing: Whether to use activation checkpointing or not.
        kwargs: Keyword arguments passed to :class:`MultiheadSelfAttention`.
    """

    def __init__(
        self,
        channels: int,
        mod_features: int = 0,
        ffn_factor: int = 4,
        ffn_activation: Literal["relu", "relu2", "silu", "swiglu"] = "silu",
        dropout: float | None = None,
        checkpointing: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.checkpointing = checkpointing

        # Ada-Norm Zero
        if hasattr(torch.nn, "RMSNorm"):
            self.norm = torch.nn.RMSNorm(channels, elementwise_affine=False, eps=1e-5)
        else:
            self.norm = RMSNorm(dim=-1, eps=1e-5)

        if mod_features > 0:
            self.ada_zero = torch.nn.Sequential(
                torch.nn.Linear(mod_features, mod_features),
                torch.nn.SiLU(),
                torch.nn.Linear(mod_features, 3 * channels),
                Rearrange("... (n C) -> n ... 1 C", n=3),
            )

            self.ada_zero[-2].weight.data.mul_(1e-2)
        else:
            self.ada_zero = torch.nn.Parameter(torch.randn(3, channels))
            self.ada_zero.data.mul_(1e-2)

        # MSA
        self.msa = MultiheadSelfAttention(channels, **kwargs)

        # FFN
        activation_factor = 1

        if ffn_activation == "relu":
            activation = torch.nn.ReLU()
        elif ffn_activation == "relu2":
            activation = ReLU2()
        elif ffn_activation == "silu":
            activation = torch.nn.SiLU()
        elif ffn_activation == "swiglu":
            activation = SwiGLU()
            activation_factor = 2
        else:
            raise NotImplementedError(f"Unknown activation '{ffn_activation}'.")

        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(channels, ffn_factor * channels),
            activation,
            torch.nn.Identity() if dropout is None else torch.nn.Dropout(dropout),
            torch.nn.Linear(ffn_factor * channels // activation_factor, channels),
        )

    def _forward(
        self,
        x: Tensor,
        mod: Tensor | None = None,
        pos: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> Tensor:
        if torch.is_tensor(self.ada_zero):
            a, b, c = self.ada_zero
        else:
            a, b, c = self.ada_zero(mod)

        y = (a + 1) * self.norm(x) + b
        y = y + self.msa(y, pos, mask)
        y = self.ffn(y)
        y = x + c * y

        return y

    def forward(
        self,
        x: Tensor,
        mod: Tensor | None = None,
        pos: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> Tensor:
        r"""
        Arguments:
            x: The input tokens :math:`x`, with shape :math:`(*, L, C)`.
            mod: The modulation vector, with shape :math:`(D)` or :math:`(*, D)`.
            pos: The postition coordinates, with shape :math:`(*, L, N)`.
            mask: The attention mask, with shape :math:`(*, L, L)`.

        Returns:
            The ouput tokens :math:`y`, with shape :math:`(*, L, C)`.
        """
        if self.checkpointing:
            return checkpoint(self._forward, reentrant=not self.training)(x, mod, pos, mask)
        else:
            return self._forward(x, mod, pos, mask)


class DiT(torch.nn.Module):
    r"""Creates a modulated DiT-like module.

    Arguments:
        in_channels: The number of input channels :math:`C_i`.
        out_channels: The number of output channels :math:`C_o`.
        cond_channels: The number of condition channels :math:`C_c`.
        mod_features: The number of modulating features :math:`D`.
        pos_channels: The number of positional channels :math:`P`.
        hid_channels: The numbers of hidden token channels :math:`C_h`.
        hid_blocks: The number of hidden transformer blocks.
        kwargs: Keyword arguments passed to :class:`DiTBlock`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_channels: int = 0,
        mod_features: int = 0,
        pos_channels: int = 1,
        hid_channels: int = 1024,
        hid_blocks: int = 3,
        **kwargs,
    ) -> None:
        super().__init__()

        self.in_proj = torch.nn.Linear(in_channels + cond_channels, hid_channels)
        self.out_proj = torch.nn.Linear(hid_channels, out_channels)

        self.pos_embedding = torch.nn.Sequential(
            SineEncoding(hid_channels, omega=1e2),
            Rearrange("... P C -> ... (P C)"),
            torch.nn.Linear(pos_channels * hid_channels, hid_channels, bias=False),
        )
        self.pos_embedding[-1].weight.data.mul_(1e-2)

        self.blocks = torch.nn.ModuleList([
            DiTBlock(
                channels=hid_channels,
                pos_channels=pos_channels,
                mod_features=mod_features,
                **kwargs,
            )
            for _ in range(hid_blocks)
        ])

    def forward(
        self,
        x: Tensor,
        mod: Tensor | None = None,
        pos: Tensor | None = None,
        cond: Tensor | None = None,
    ) -> Tensor:
        r"""
        Arguments:
            x: The input tensor, with shape :math:`(*, L, C_i)`.
            mod: The modulation vector, with shape :math:`(D)` or :math:`(*, D)`.
            pos: The position tensor, with shape :math:`(*, L, P)`.
                If `None`, use the sequence indices instead.
            cond: The condition tensor, with shape :math:`(*, L, C_c)`.

        Returns:
            The output tensor, with shape :math:`(*, L, C_o)`.
        """
        if cond is not None:
            x = torch.cat((x, cond), dim=-1)

        x = self.in_proj(x)

        if pos is None:
            pos = torch.arange(x.shape[-2], dtype=x.dtype, device=x.device)
            pos = pos[..., None]

        x = x + self.pos_embedding(pos)

        for block in self.blocks:
            x = block(x, mod, pos=pos)

        x = self.out_proj(x)

        return x
