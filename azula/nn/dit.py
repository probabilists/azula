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
                torch.nn.Linear(mod_features, 6 * channels),
                Rearrange("... (n C) -> n ... 1 C", n=6),
            )
            self.ada_zero[0].weight.data.mul_(1e-2)
            self.ada_zero[0].bias.data.zero_()
        else:
            self.ada_zero = None

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
            torch.nn.Linear(channels, ffn_factor * activation_factor * channels),
            activation,
            torch.nn.Identity() if dropout is None else torch.nn.Dropout(dropout),
            torch.nn.Linear(ffn_factor * channels, channels),
        )

        if self.ada_zero is None:
            self.msa.y_proj.weight.data.mul_(1e-2)
            self.ffn[-1].weight.data.mul_(1e-2)
            self.ffn[-1].bias.data.zero_()

    def _forward(
        self,
        x: Tensor,
        mod: Tensor | None = None,
        pos: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> Tensor:
        if self.ada_zero is None:
            x = x + self.msa(self.norm(x), pos, mask)
            x = x + self.ffn(self.norm(x))
        else:
            a1, b1, c1, a2, b2, c2 = self.ada_zero(mod)

            x = x + c1 * self.msa((a1 + 1) * self.norm(x) + b1, pos, mask)
            x = x + c2 * self.ffn((a2 + 1) * self.norm(x) + b2)

        return x

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
            mod: The modulation vector, with shape :math:`(*, D)`.
            pos: The postition coordinates, with shape :math:`(*, L, N)`.
            mask: The attention mask, broadcasting with shape :math:`(*, H, L, L)`.

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
        ape: Whether to use absolute positional embedding (APE) or not.
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
        ape: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        self.in_proj = torch.nn.Linear(in_channels + cond_channels, hid_channels)
        self.out_proj = torch.nn.Linear(hid_channels, out_channels)

        if hasattr(torch.nn, "RMSNorm"):
            self.out_norm = torch.nn.RMSNorm(hid_channels, elementwise_affine=False, eps=1e-5)
        else:
            self.out_norm = RMSNorm(dim=-1, eps=1e-5)

        if mod_features > 0:
            self.out_ada_zero = torch.nn.Sequential(
                torch.nn.Linear(mod_features, 2 * hid_channels),
                Rearrange("... (n C) -> n ... 1 C", n=2),
            )
            self.out_ada_zero[0].weight.data.mul_(1e-2)
            self.out_ada_zero[0].bias.data.zero_()
        else:
            self.out_ada_zero = None

        if ape:
            self.pos_embedding = torch.nn.Sequential(
                SineEncoding(hid_channels, omega=1e2),
                Rearrange("... P C -> ... (P C)"),
                torch.nn.Linear(pos_channels * hid_channels, hid_channels, bias=False),
            )
            self.pos_embedding[-1].weight.data.mul_(1e-2)
        else:
            self.pos_embedding = None

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
        mask: Tensor | None = None,
        return_hidden: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        r"""
        Arguments:
            x: The input tensor, with shape :math:`(*, L, C_i)`.
            mod: The modulation vector, with shape :math:`(*, D)`.
            pos: The position tensor, with shape :math:`(*, L, P)`.
                If `None`, use the sequence indices instead.
            cond: The condition tensor, with shape :math:`(*, L, C_c)`.
            mask: The attention mask, broadcasting with shape :math:`(*, H, L, L)`.
            return_hidden: Whether to return the trunk's final hidden state or not.

        Returns:
            The output tensor, with shape :math:`(*, L, C_o)`. If `return_hidden=True`,
            also returns the hidden state, with shape :math:`(*, L, C_h)`.
        """
        *_, L, _ = x.shape

        if cond is not None:
            x = torch.cat((x, cond), dim=-1)

        x = self.in_proj(x)

        if pos is None:
            pos = torch.arange(L, dtype=x.dtype, device=x.device)
            pos = pos[..., None]

        if self.pos_embedding is not None:
            x = x + self.pos_embedding(pos)

        h = x
        for block in self.blocks:
            h = block(h, mod, pos=pos, mask=mask)

        if self.out_ada_zero is None:
            x = self.out_norm(h)
        else:
            a, b = self.out_ada_zero(mod)
            x = (a + 1) * self.out_norm(h) + b
        x = self.out_proj(x)

        if return_hidden:
            return x, h
        else:
            return x
