r"""Conditional sampling and guidance."""

__all__ = [
    "MMPSDenoiser",
]

import torch

from functools import partial
from torch import Tensor
from typing import Callable

# isort: split
from ..denoise import Gaussian, GaussianDenoiser
from ..linalg.solve import cg, gmres


class MMPSDenoiser(GaussianDenoiser):
    r"""Creates a Moment Matching Posterior Sampling (MMPS) denoiser module.

    References:
        | Learning Diffusion Priors from Observations by Expectation Maximization (Rozet et al., 2024)
        | https://arxiv.org/abs/2405.13712

    Arguments:
        denoiser: A Gaussian denoiser.
        y: An observation :math:`y \sim \mathcal{N}(Ax, \Sigma_y)`.
        A: The forward operator :math:`x \mapsto Ax`.
        var_y: The noise variance :math:`\Sigma_y`.
    """

    def __init__(
        self,
        denoiser: GaussianDenoiser,
        y: Tensor,
        A: Callable[[Tensor], Tensor],
        var_y: Tensor,
        solver: str = "gmres",
        iterations: int = 1,
    ):
        super().__init__(schedule=denoiser.schedule)

        self.denoiser = denoiser

        self.A = A

        self.register_buffer("y", torch.as_tensor(y))
        self.register_buffer("var_y", torch.as_tensor(var_y))

        if solver == "cg":
            self.solve = partial(cg, iterations=iterations)
        elif solver == "gmres":
            self.solve = partial(gmres, iterations=iterations)
        else:
            raise ValueError(f"Unknown solver '{solver}'.")

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

        def cov_y(v):
            return self.var_y * v + self.A(cov_x(At(v)))

        grad = self.y - y_hat
        grad = self.solve(A=cov_y, b=grad)
        grad = torch.autograd.grad(y_hat, x_t, grad)[0]

        return Gaussian(
            mean=x_hat + gamma_t * grad,
            var=q.var,
        )
