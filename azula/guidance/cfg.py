r"""Classifier-free guidance (CFG) internals.

References:
    | Classifier-Free Diffusion Guidance (Ho et al., 2022)
    | https://arxiv.org/abs/2207.12598
"""

__all__ = [
    "CFGDenoiser",
]

from torch import Tensor
from typing import Any, Dict, Union

from ..denoise import Denoiser, DiracPosterior
from ..noise import Schedule


class CFGDenoiser(Denoiser):
    r"""Creates a CFG denoiser module.

    Arguments:
        denoiser: A denoiser :math:`q_\phi(X \mid X_t)`.
    """

    def __init__(self, denoiser: Denoiser):
        super().__init__()

        self.denoiser = denoiser

    @property
    def schedule(self) -> Schedule:
        return self.denoiser.schedule

    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        positive: Dict[str, Any],
        negative: Dict[str, Any] = {},  # noqa: B006
        guidance: Union[float, Tensor] = 1.0,
        **kwargs,
    ) -> DiracPosterior:
        r"""
        Arguments:
            x_t: A noisy tensor :math:`x_t`, with shape :math:`(B, *)`.
            t: The time :math:`t`, with shape :math:`()` or :math:`(B)`.
            positive: The positive label :math:`c_+` as a dictionary of keyword arguments.
            negative: The negative label :math:`c_-` as a dictionary of keyword arguments.
            guidance: The classifier-free guidance strength :math:`\omega \in \mathbb{R}_+`.
            kwargs: Optional keyword arguments.

        Returns:
            The Dirac delta :math:`\delta(X - \mu)` where

            .. math:: \mu = (1 + \omega) \, \mu_\phi(x_t \mid c_+)
                - \omega \, \mu_\phi(x_t \mid c_-) \, .
        """

        q_pos = self.denoiser(x_t, t, **positive, **kwargs)
        q_neg = self.denoiser(x_t, t, **negative, **kwargs)

        return DiracPosterior(
            mean=q_pos.mean + guidance * (q_pos.mean - q_neg.mean),
        )
