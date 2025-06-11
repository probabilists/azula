r"""Tweedie Moment Projected Diffusion (TMPD) internals.

References:
    | Tweedie Moment Projected Diffusions For Inverse Problems (Boys et al., 2023)
    | https://arxiv.org/abs/2310.06721
"""

__all__ = [
    "TMPDenoiser",
]

import torch

from torch import Tensor
from typing import Callable

from ..denoise import Gaussian, GaussianDenoiser
from ..noise import Schedule


class TMPDenoiser(GaussianDenoiser):
    r"""Creates a TMPD denoiser module.

    Arguments:
        denoiser: A Gaussian denoiser.
        y: An observation :math:`y \sim \mathcal{N}(A x, \Sigma_y)`.
        A: The forward operator :math:`x \mapsto A x`.
        var_y: The noise variance :math:`\Sigma_y`.
    """

    def __init__(
        self,
        denoiser: GaussianDenoiser,
        y: Tensor,
        A: Callable[[Tensor], Tensor],
        var_y: Tensor,
    ):
        super().__init__()

        self.denoiser = denoiser

        self.A = A

        self.register_buffer("y", torch.as_tensor(y))
        self.register_buffer("var_y", torch.as_tensor(var_y))

    @property
    def schedule(self) -> Schedule:
        return self.denoiser.schedule

    def forward(self, x_t: Tensor, t: Tensor, **kwargs) -> Gaussian:
        alpha_t, sigma_t = self.schedule(t)
        gamma_t = sigma_t**2 / alpha_t

        with torch.enable_grad():
            x_t = x_t.detach().requires_grad_()
            q = self.denoiser(x_t, t, **kwargs)

            x_hat = q.mean
            y_hat = self.A(x_hat)

        def At(v):
            return torch.autograd.grad(y_hat, x_hat, v, retain_graph=True)[0]

        def cov_x(v):
            return gamma_t * torch.autograd.grad(x_hat, x_t, v, retain_graph=True)[0]

        var_Ax = self.A(cov_x(At(torch.ones_like(y_hat))))

        grad = (self.y - y_hat) / (self.var_y + var_Ax)
        grad = torch.autograd.grad(y_hat, x_t, grad)[0]

        return Gaussian(
            mean=x_hat + gamma_t * grad,
            var=q.var,
        )
