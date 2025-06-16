r"""Posterior Sampling with Latent Diffusion (PSLD) internals.

References:
    | Solving Linear Inverse Problems Provably via Posterior Sampling with Latent Diffusion Models (Rout et al., 2023)
    | https://arxiv.org/abs/2307.00619
"""

__all__ = [
    "PSLDSampler",
]

import torch

from torch import Tensor
from typing import Callable

from ..denoise import GaussianDenoiser
from ..sample import DDIMSampler


class PSLDSampler(DDIMSampler):
    r"""Creates a PSLD sampler.

    Arguments:
        denoiser: A Gaussian denoiser.
        y: An observation :math:`y \sim \mathcal{N}(A x, \Sigma_y)`.
        A: The forward operator :math:`x \mapsto A x`.
        E: The encoder :math:`x \mapsto E(x) = z`.
        D: The decoder :math:`z \mapsto D(z) = x`.
        zeta: The guidance strength :math:`\zeta`.
        gamma: The goodness strength :math:`\gamma`.
        kwargs: Keyword arguments passed to :class:`azula.sample.DDIMSampler`.
    """

    def __init__(
        self,
        denoiser: GaussianDenoiser,
        y: Tensor,
        A: Callable[[Tensor], Tensor],
        E: Callable[[Tensor], Tensor],
        D: Callable[[Tensor], Tensor],
        zeta: float = 1.0,
        gamma: float = 1.0,
        **kwargs,
    ):
        super().__init__(denoiser, **kwargs)

        self.A = A
        self.E = E
        self.D = D

        self.register_buffer("y", torch.as_tensor(y))
        self.register_buffer("zeta", torch.as_tensor(zeta))
        self.register_buffer("gamma", torch.as_tensor(gamma))

    def step(self, z_t: Tensor, t: Tensor, s: Tensor, **kwargs) -> Tensor:
        # DDIM
        alpha_s, sigma_s = self.denoiser.schedule(s)
        alpha_t, sigma_t = self.denoiser.schedule(t)

        tau = 1 - (alpha_t / alpha_s * sigma_s / sigma_t) ** 2
        tau = torch.clip(self.eta * tau, min=0, max=1)
        eps = torch.randn_like(z_t)

        with torch.enable_grad():
            z_t = z_t.detach().requires_grad_()
            z_hat = self.denoiser(z_t, t, **kwargs).mean

        z_s = alpha_s * z_hat
        z_s = z_s + sigma_s * torch.sqrt(1 - tau) / sigma_t * (z_t - alpha_t * z_hat)
        z_s = z_s + sigma_s * torch.sqrt(tau) * eps

        # PSLD
        with torch.enable_grad():
            x_hat = self.D(z_hat)
            y_hat = self.A(x_hat)

            def At(v):
                return torch.autograd.grad(y_hat, x_hat, v, retain_graph=True)[0]

            error = torch.linalg.norm(self.y - y_hat)
            goodness = torch.linalg.norm(self.E(x_hat + At(self.y - y_hat)) - z_hat)
            loss = self.zeta * error + self.gamma * goodness

        grad = torch.autograd.grad(loss, z_t)[0]

        return z_s - grad
