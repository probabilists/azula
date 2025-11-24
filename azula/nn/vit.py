r"""Vision Transformer (ViT) building blocks.

References:
    | An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (Dosovitskiy et al., 2021)
    | https://arxiv.org/abs/2010.11929

    | Scalable Diffusion Models with Transformers (Peebles et al., 2022)
    | https://arxiv.org/abs/2212.09748
"""

__all__ = [
    "ViTBlock",
    "ViT",
]

import math
import torch
import torch.nn as nn

from einops.layers.torch import Rearrange
from torch import Tensor
from typing import Optional, Sequence, Union

from .attention import MultiheadSelfAttention
from .embedding import SineEncoding
from .layers import Patchify, ReLU2, SwiGLU, Unpatchify
from .utils import checkpoint


class ViTBlock(nn.Module):
    r"""Creates a ViT block module.

    Arguments:
        channels: The number of channels :math:`C`.
        mod_features: The number of modulating features :math:`D`.
        ffn_factor: The channel factor in the FFN.
        ffn_activation: The activation function in the FFN. Options are `relu`, `relu2`,
            `silu` and `swiglu`.
        dropout: The dropout rate in :math:`[0, 1]`.
        checkpointing: Whether to use gradient checkpointing or not.
        kwargs: Keyword arguments passed to :class:`azula.nn.attention.MultiheadSelfAttention`.
    """

    def __init__(
        self,
        channels: int,
        mod_features: int = 0,
        ffn_factor: int = 4,
        ffn_activation: str = "silu",
        dropout: Optional[float] = None,
        checkpointing: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.checkpointing = checkpointing

        # Ada-LN Zero
        self.norm = nn.LayerNorm(channels, elementwise_affine=False)

        if mod_features > 0:
            self.ada_zero = nn.Sequential(
                nn.Linear(mod_features, mod_features),
                nn.SiLU(),
                nn.Linear(mod_features, 3 * channels),
                Rearrange("... (n C) -> n ... 1 C", n=3),
            )

            self.ada_zero[-2].weight.data.mul_(1e-2)
        else:
            self.ada_zero = nn.Parameter(torch.randn(3, channels))
            self.ada_zero.data.mul_(1e-2)

        # MSA
        self.msa = MultiheadSelfAttention(channels, **kwargs)

        # FFN
        activation_factor = 1

        if ffn_activation == "relu":
            activation = nn.ReLU()
        elif ffn_activation == "relu2":
            activation = ReLU2()
        elif ffn_activation == "silu":
            activation = nn.SiLU()
        elif ffn_activation == "swiglu":
            activation = SwiGLU()
            activation_factor = 2
        else:
            raise ValueError(f"Unknown activation '{ffn_activation}'.")

        self.ffn = nn.Sequential(
            nn.Linear(channels, ffn_factor * channels),
            activation,
            nn.Identity() if dropout is None else nn.Dropout(dropout),
            nn.Linear(ffn_factor * channels // activation_factor, channels),
        )

    def _forward(
        self,
        x: Tensor,
        mod: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        if torch.is_tensor(self.ada_zero):
            a, b, c = self.ada_zero
        else:
            a, b, c = self.ada_zero(mod)

        y = (a + 1) * self.norm(x) + b
        y = y + self.msa(y, pos, mask)
        y = self.ffn(y)
        y = (x + c * y) * torch.rsqrt(1 + c * c)

        return y

    def forward(
        self,
        x: Tensor,
        mod: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
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


class ViT(nn.Module):
    r"""Creates a modulated ViT-like module.

    Arguments:
        in_channels: The number of input channels :math:`C_i`.
        out_channels: The number of output channels :math:`C_o`.
        cond_channels: The number of condition channels :math:`C_c`.
        mod_features: The number of modulating features :math:`D`.
        hid_channels: The numbers of hidden token channels.
        hid_blocks: The number of hidden transformer blocks.
        spatial: The number of spatial dimensions :math:`N`.
        patch_size: The patch size or shape.
        unpatch_size: The unpatch size or shape.
        kwargs: Keyword arguments passed to :class:`ViTBlock`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_channels: int = 0,
        mod_features: int = 0,
        hid_channels: int = 1024,
        hid_blocks: int = 3,
        spatial: int = 2,
        patch_size: Union[int, Sequence[int]] = 1,
        unpatch_size: Union[int, Sequence[int], None] = None,
        **kwargs,
    ):
        super().__init__()

        kwargs.setdefault("rpb", True)

        if isinstance(patch_size, int):
            patch_size = [patch_size] * spatial

        if unpatch_size is None:
            unpatch_size = patch_size
        elif isinstance(unpatch_size, int):
            unpatch_size = [unpatch_size] * spatial

        assert len(patch_size) == len(unpatch_size) == spatial

        self.patch = Patchify(patch_size, channel_last=True)
        self.unpatch = Unpatchify(unpatch_size, channel_last=True)
        self.spatial = spatial

        self.in_proj = nn.Linear(math.prod(patch_size) * (in_channels + cond_channels), hid_channels)  # fmt: off
        self.out_proj = nn.Linear(hid_channels, math.prod(patch_size) * out_channels)

        self.pos_embedding = nn.Sequential(
            SineEncoding(hid_channels, omega=1e2),
            Rearrange("... N C -> ... (N C)"),
            nn.Linear(spatial * hid_channels, hid_channels, bias=False),
        )
        self.pos_embedding[-1].weight.data.mul_(1e-2)

        self.blocks = nn.ModuleList([
            ViTBlock(
                channels=hid_channels,
                mod_features=mod_features,
                pos_features=spatial,
                **kwargs,
            )
            for _ in range(hid_blocks)
        ])

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
            cond: The condition tensor, with :math:`(B, C_c, L_1, ..., L_N)`.

        Returns:
            The output tensor, with shape :math:`(B, C_o, L_1, ..., L_N)`.
        """

        if cond is not None:
            x = torch.cat((x, cond), dim=1)

        x = self.patch(x)
        x = self.in_proj(x)

        shape = x.shape[1:-1]

        pos = (torch.arange(size, dtype=x.dtype, device=x.device) for size in shape)
        pos = torch.cartesian_prod(*pos)
        pos = torch.reshape(pos, shape=(-1, len(shape)))

        x = torch.flatten(x, 1, -2)
        x = x + self.pos_embedding(pos)

        for block in self.blocks:
            x = block(x, mod, pos=pos)

        x = torch.unflatten(x, sizes=shape, dim=-2)

        x = self.out_proj(x)
        x = self.unpatch(x)

        return x
