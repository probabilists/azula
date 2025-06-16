r"""Pseudo-inverse Guided Diffusion Model (PGDM) internals.

References:
    | Pseudoinverse-Guided Diffusion Models for Inverse Problems (Song et al., 2023)
    | https://openreview.net/forum?id=9_gsMA8MRKQ
"""

__all__ = [
    "PGDMSampler",
]

import torch

from torch import Tensor
from typing import Callable

from ..denoise import GaussianDenoiser
from ..sample import DDIMSampler


class PGDMSampler(DDIMSampler):
    r"""Creates a PGDM sampler.

    Arguments:
        denoiser: A Gaussian denoiser.
        y: An observation :math:`y \sim \mathcal{N}(A(x), \Sigma_y)`.
        A: The forward operator :math:`x \mapsto A(x)`.
        A_inv: The pseudo-inverse operator :math:`y \mapsto A^\dagger(y)`,
            such that :math:`A(A^\dagger(A(x))) = A(x)`.
        kwargs: Keyword arguments passed to :class:`azula.sample.DDIMSampler`.
    """

    def __init__(
        self,
        denoiser: GaussianDenoiser,
        y: Tensor,
        A: Callable[[Tensor], Tensor],
        A_inv: Callable[[Tensor], Tensor],
        **kwargs,
    ):
        super().__init__(denoiser, **kwargs)

        self.A = A
        self.A_inv = A_inv

        self.register_buffer("y", torch.as_tensor(y))

    def step(self, x_t: Tensor, t: Tensor, s: Tensor, **kwargs) -> Tensor:
        # DDIM
        alpha_s, sigma_s = self.denoiser.schedule(s)
        alpha_t, sigma_t = self.denoiser.schedule(t)

        tau = 1 - (alpha_t / alpha_s * sigma_s / sigma_t) ** 2
        tau = torch.clip(self.eta * tau, min=0, max=1)
        eps = torch.randn_like(x_t)

        with torch.enable_grad():
            x_t = x_t.detach().requires_grad_()
            x_hat = self.denoiser(x_t, t, **kwargs).mean

        x_s = alpha_s * x_hat
        x_s = x_s + sigma_s * torch.sqrt(1 - tau) / sigma_t * (x_t - alpha_t * x_hat)
        x_s = x_s + sigma_s * torch.sqrt(tau) * eps

        # PiGDM
        grad = self.A_inv(self.y) - self.A_inv(self.A(x_hat))
        grad = torch.autograd.grad(x_hat, x_t, grad)[0]

        return x_s + alpha_s * alpha_t * grad
