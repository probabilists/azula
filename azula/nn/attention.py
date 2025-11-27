r"""Attention layers."""

__all__ = [
    "MultiheadSelfAttention",
]

import math
import torch
import torch.nn as nn

from einops import rearrange
from torch import BoolTensor, Tensor
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple

from .layers import RMSNorm
from .utils import promote_dtype


class MultiheadSelfAttention(nn.Module):
    r"""Creates a multi-head self-attention layer.

    Arguments:
        channels: The number of channels :math:`H \times C`.
        pos_channels: The number of positional channels :math:`P`.
            Only necessary with RoPE.
        attention_heads: The number of attention heads :math:`H`.
        qkv_bias: Whether to add bias to the query-key-value projection layer or not.
        qk_norm: Whether to use query-key RMS-normalization or not.
        rope: Whether to use rotary positional embedding (RoPE) or not.
        dropout: The dropout rate in :math:`[0, 1]`.
        checkpointing: Whether to use activation checkpointing or not.
    """

    def __init__(
        self,
        channels: int,
        pos_channels: int = 1,
        attention_heads: int = 1,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        rope: bool = False,
        dropout: Optional[float] = None,
        checkpointing: bool = False,
    ):
        super().__init__()

        assert channels % attention_heads == 0

        self.qkv_proj = nn.Linear(channels, 3 * channels, bias=qkv_bias)
        self.y_proj = nn.Linear(channels, channels, bias=False)

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

        if rope:
            magnitude = torch.exp(math.log(1e-1) * torch.rand(channels // 2, 1))
            direction = torch.randn(channels // 2, pos_channels)
            direction = direction / torch.linalg.norm(direction, dim=-1, keepdim=True)

            self.theta_proj = nn.Linear(pos_channels, channels // 2, bias=False)
            self.theta_proj.weight.data.copy_(magnitude * direction)
        else:
            self.theta_proj = None

        self.heads = attention_heads
        self.dropout = 0.0 if dropout is None else dropout
        self.checkpointing = checkpointing

    def _forward(
        self,
        x: Tensor,
        pos: Optional[Tensor] = None,
        mask: Optional[BoolTensor] = None,
    ) -> Tensor:
        qkv = self.qkv_proj(x)
        q, k, v = rearrange(qkv, "... L (n H C) -> n ... H L C", n=3, H=self.heads)
        q, k = self.qk_norm(q), self.qk_norm(k)

        if self.theta_proj is not None:
            theta = self.theta_proj(pos)
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
            pos: Optional position vectors :math:`p`, with shape :math:`(*, L, P)`.
            mask: Optional attention mask, with shape :math:`(L, L)`.

        Returns:
            The ouput tokens :math:`y`, with shape :math:`(*, L, H \times C)`.
        """

        if self.checkpointing:
            return checkpoint(self._forward, reentrant=not self.training)(x, theta, mask)
        else:
            return self._forward(x, theta, mask)


@promote_dtype
def apply_rope(q: Tensor, k: Tensor, theta: Tensor) -> Tuple[Tensor, Tensor]:
    r"""
    References:
        | RoFormer: Enhanced Transformer with Rotary Position Embedding (Su et al., 2021)
        | https://arxiv.org/abs/2104.09864

        | Rotary Position Embedding for Vision Transformer (Heo et al., 2024)
        | https://arxiv.org/abs/2403.13298

    Arguments:
        q: The query vectors :math:`q`, with shape :math:`(*, C)`.
        k: The key vectors :math:`k`, with shape :math:`(*, C)`.
        theta: Rotary angles, with shape :math:`(*, C / 2)`.

    Returns:
        The rotated query and key vectors, with shape :math:`(*, C)`.
    """

    q = q.unflatten(-1, (-1, 2))
    k = k.unflatten(-1, (-1, 2))

    q_real, q_imag = q[..., 0], q[..., 1]
    k_real, k_imag = k[..., 0], k[..., 1]

    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    q = torch.stack(
        (
            q_real * cos_theta - q_imag * sin_theta,
            q_real * sin_theta + q_imag * cos_theta,
        ),
        dim=-1,
    ).flatten(-2)

    k = torch.stack(
        (
            k_real * cos_theta - k_imag * sin_theta,
            k_real * sin_theta + k_imag * cos_theta,
        ),
        dim=-1,
    ).flatten(-2)

    return q, k
