r"""Vision Transformer (ViT) building blocks.

References:
    | An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (Dosovitskiy et al., 2021)
    | https://arxiv.org/abs/2010.11929
"""

__all__ = [
    "ViT",
]

import math
import torch

from collections.abc import Sequence
from torch import Tensor

from .dit import DiT
from .layers import Patchify, Unpatchify


class ViT(DiT):
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
        patch_size: int | Sequence[int] = 1,
        unpatch_size: int | Sequence[int] | None = None,
        **kwargs,
    ) -> None:
        if isinstance(patch_size, int):
            patch_size = [patch_size] * spatial

        if unpatch_size is None:
            unpatch_size = patch_size
        elif isinstance(unpatch_size, int):
            unpatch_size = [unpatch_size] * spatial

        assert len(patch_size) == len(unpatch_size) == spatial

        super().__init__(
            in_channels=math.prod(patch_size) * in_channels,
            out_channels=math.prod(unpatch_size) * out_channels,
            cond_channels=math.prod(patch_size) * cond_channels,
            mod_features=mod_features,
            pos_channels=spatial,
            hid_channels=hid_channels,
            hid_blocks=hid_blocks,
            **kwargs,
        )

        self.patch = Patchify(patch_size, channel_last=True)
        self.unpatch = Unpatchify(unpatch_size, channel_last=True)
        self.spatial = spatial

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
            cond: The condition tensor, with :math:`(B, C_c, L_1, ..., L_N)`.

        Returns:
            The output tensor, with shape :math:`(B, C_o, L_1, ..., L_N)`.
        """

        x = self.patch(x)

        if cond is not None:
            cond = self.patch(cond)

        shape = x.shape[1:-1]

        pos = (torch.arange(size, dtype=x.dtype, device=x.device) for size in shape)
        pos = torch.cartesian_prod(*pos)
        pos = torch.reshape(pos, shape=(-1, len(shape)))

        x = x.flatten(1, -2)
        y = super().forward(x, mod, pos=pos, cond=cond)
        y = y.unflatten(-2, shape)
        y = self.unpatch(y)

        return y
