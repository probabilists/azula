r"""Attention layers."""

__all__ = [
    "MultiheadSelfAttention",
]

import torch
import torch.nn as nn

from einops import rearrange
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple

from .layers import RMSNorm


class MultiheadSelfAttention(nn.Module):
    r"""Creates a multi-head self-attention layer.

    Arguments:
        channels: The number of channels :math:`H \times C`.
        attention_heads: The number of attention heads :math:`H`.
        qk_norm: Whether to use query-key RMS-normalization or not.
        dropout: The dropout rate in :math:`[0, 1]`.
        checkpointing: Whether to use gradient checkpointing or not.
    """

    def __init__(
        self,
        channels: int,
        attention_heads: int = 1,
        qk_norm: bool = True,
        dropout: Optional[float] = None,
        checkpointing: bool = False,
    ):
        super().__init__()

        assert channels % attention_heads == 0

        self.qkv_proj = nn.Linear(channels, 3 * channels, bias=False)
        self.y_proj = nn.Linear(channels, channels)

        if qk_norm:
            if hasattr(nn, "RMSNorm"):
                self.qk_norm = nn.RMSNorm(
                    channels // attention_heads,
                    elementwise_affine=False,
                    eps=1e-5,
                )
            else:
                self.qk_norm = RMSNorm(dim=-1, eps=1e-5)
        else:
            self.qk_norm = nn.Identity()

        self.heads = attention_heads
        self.dropout = 0.0 if dropout is None else dropout
        self.checkpointing = checkpointing

    def _forward(
        self,
        x: Tensor,
        theta: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        qkv = self.qkv_proj(x)
        q, k, v = rearrange(qkv, "... L (n H C) -> n ... H L C", n=3, H=self.heads)
        q, k = self.qk_norm(q), self.qk_norm(k)

        if theta is not None:
            theta = rearrange(theta, "... L (H C) -> ... H L C", H=self.heads)
            q, k = apply_rope(q, k, theta)

        y = torch.nn.functional.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
        )

        y = rearrange(y, "... H L C -> ... L (H C)")
        y = self.y_proj(y)

        return y

    def forward(
        self,
        x: Tensor,
        theta: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""
        Arguments:
            x: The input tokens :math:`x`, with shape :math:`(*, L, H \times C)`.
            theta: Optional rotary positional embedding :math:`\theta`,
                with shape :math:`(*, L, H \times C / 2)`.
            mask: Optional attention mask, with shape :math:`(L, L)`.

        Returns:
            The ouput tokens :math:`y`, with shape :math:`(*, L, H \times C)`.
        """

        if self.checkpointing:
            return checkpoint(self._forward, x, theta, mask, use_reentrant=False)
        else:
            return self._forward(x, theta, mask)


def apply_rope(q: Tensor, k: Tensor, theta: Tensor) -> Tuple[Tensor, Tensor]:
    r"""
    References:
        | RoFormer: Enhanced Transformer with Rotary Position Embedding (Su et al., 2021)
        | https://arxiv.org/abs/2104.09864

        | Rotary Position Embedding for Vision Transformer (Heo et al., 2024)
        | https://arxiv.org/abs/2403.13298

    Arguments:
        q: The query tokens :math:`q`, with shape :math:`(*, C)`.
        k: The key tokens :math:`k`, with shape :math:`(*, C)`.
        theta: Rotary angles, with shape :math:`(*, C / 2)`.

    Returns:
        The rotated query and key tokens, with shape :math:`(*, C)`.
    """

    rotation = torch.polar(torch.ones_like(theta), theta)

    q = torch.view_as_complex(torch.unflatten(q, -1, (-1, 2)))
    k = torch.view_as_complex(torch.unflatten(k, -1, (-1, 2)))

    q = torch.flatten(torch.view_as_real(rotation * q), -2)
    k = torch.flatten(torch.view_as_real(rotation * k), -2)

    return q, k
