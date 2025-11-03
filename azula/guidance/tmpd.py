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
from typing import Callable, Union

from ..denoise import Denoiser, DiracPosterior
from ..noise import Schedule


class TMPDenoiser(Denoiser):
    r"""Creates a TMPD denoiser module.

    Arguments:
        denoiser: A denoiser :math:`q_\phi(X \mid X_t)`.
        y: An observation :math:`y \sim \mathcal{N}(A x, \Sigma_y)`.
        A: The forward operator :math:`x \mapsto A x`.
        var_y: The noise variance :math:`\Sigma_y`.
    """

    def __init__(
        self,
        denoiser: Denoiser,
        y: Tensor,
        A: Callable[[Tensor], Tensor],
        var_y: Union[float, Tensor],
    ):
        super().__init__()

        self.denoiser = denoiser

        self.y = y
        self.A = A
        self.var_y = var_y

    @property
    def schedule(self) -> Schedule:
        return self.denoiser.schedule

    def forward(self, x_t: Tensor, t: Tensor, **kwargs) -> DiracPosterior:
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

        return DiracPosterior(mean=x_hat + gamma_t * grad)
