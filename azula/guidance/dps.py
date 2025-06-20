r"""Diffusion Posterior Sampling (DPS) internals.

References:
    | Diffusion Posterior Sampling for General Noisy Inverse Problems (Chung et al., 2022)
    | https://arxiv.org/abs/2209.14687
"""

__all__ = [
    "DPSSampler",
]

import torch

from torch import Tensor
from typing import Callable

from ..denoise import GaussianDenoiser
from ..sample import Sampler


class DPSSampler(Sampler):
    r"""Creates a DPS sampler.

    Arguments:
        denoiser: A Gaussian denoiser.
        y: An observation :math:`y \sim \mathcal{N}(A(x), \Sigma_y)`.
        A: The forward operator :math:`x \mapsto A(x)`.
        zeta: The guidance strength :math:`\zeta`.
        kwargs: Keyword arguments passed to :class:`azula.sample.Sampler`.
    """

    def __init__(
        self,
        denoiser: GaussianDenoiser,
        y: Tensor,
        A: Callable[[Tensor], Tensor],
        zeta: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.denoiser = denoiser
        self.A = A

        self.register_buffer("y", torch.as_tensor(y))
        self.register_buffer("zeta", torch.as_tensor(zeta))

    def step(self, x_t: Tensor, t: Tensor, s: Tensor, **kwargs) -> Tensor:
        # DDPM
        alpha_s, sigma_s = self.denoiser.schedule(s)
        alpha_t, sigma_t = self.denoiser.schedule(t)

        tau = 1 - (alpha_t / alpha_s * sigma_s / sigma_t) ** 2
        eps = torch.randn_like(x_t)

        with torch.enable_grad():
            x_t = x_t.detach().requires_grad_()
            x_hat = self.denoiser(x_t, t, **kwargs).mean

        x_s = alpha_s * x_hat
        x_s = x_s + sigma_s * torch.sqrt(1 - tau) / sigma_t * (x_t - alpha_t * x_hat)
        x_s = x_s + sigma_s * torch.sqrt(tau) * eps

        # DPS
        with torch.enable_grad():
            error = self.y - self.A(x_hat)
            norm = torch.linalg.vector_norm(error)

        grad = torch.autograd.grad(norm, x_t)[0]

        return x_s - self.zeta * grad
