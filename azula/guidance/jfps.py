r"""Jacobian-Free Posterior Sampling (JFPS) internals."""

from __future__ import annotations

__all__ = [
    "JFPSDenoiser",
]

import torch

from functools import partial
from torch import Tensor
from typing import Callable

from ..denoise import Denoiser, DiracPosterior
from ..linalg.covariance import Covariance, IsotropicCovariance
from ..linalg.solve import cg, gmres
from ..noise import Schedule


class JFPSDenoiser(Denoiser):
    r"""Creates a JFPS denoiser module.

    Arguments:
        denoiser: A denoiser :math:`q_\phi(X \mid X_t)`.
        y: An observation :math:`y \sim \mathcal{N}(A(x), \Sigma_y)`, with shape :math:`(*, D)`.
        A: The forward operator :math:`x \mapsto A(x)`.
        cov_y: The noise covariance :math:`\Sigma_y`.
        cov_x: The signal covariance :math:`\Sigma_x`.
        solver: The linear solver name (:py:`"cg"` or :py:`"gmres"`).
        iterations: The number of solver iterations.
    """

    def __init__(
        self,
        denoiser: Denoiser,
        y: Tensor,
        A: Callable[[Tensor], Tensor],
        cov_y: Covariance,
        cov_x: Covariance,
        solver: str = "cg",
        iterations: int = 1,
    ):
        super().__init__()

        self.denoiser = denoiser

        self.A = A
        self.cov_y = cov_y
        self.cov_x = cov_x

        self.register_buffer("y", torch.as_tensor(y))

        if solver == "cg":
            self.solve = partial(cg, iterations=iterations)
        elif solver == "gmres":
            self.solve = partial(gmres, iterations=iterations)
        else:
            raise ValueError(f"Unknown solver '{solver}'.")

    @property
    def schedule(self) -> Schedule:
        return self.denoiser.schedule

    def forward(self, x_t: Tensor, t: Tensor, **kwargs) -> DiracPosterior:
        alpha_t, sigma_t = self.schedule(t)

        q = self.denoiser(x_t, t, **kwargs)

        with torch.enable_grad():
            x_hat = q.mean.detach().requires_grad_()
            y_hat = self.A(x_hat)

        def A(v):
            return torch.func.jvp(self.A, (x_hat.detach(),), (v,))[1]

        def At(v):
            return torch.autograd.grad(y_hat, x_hat, v, retain_graph=True)[0]

        cov_t = IsotropicCovariance(sigma_t**2 / alpha_t**2)
        cov_x = (self.cov_x.inv + cov_t.inv).inv

        def cov_y(v):
            return self.cov_y(v) + A(cov_x(At(v)))

        grad = self.y - y_hat
        grad = self.solve(A=cov_y, b=grad)
        grad = torch.autograd.grad(y_hat, x_hat, grad)[0]
        grad = cov_x(grad)

        return DiracPosterior(mean=x_hat + grad)
