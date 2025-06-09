r"""Moment Matching Posterior Sampling (MMPS) internals.

References:
    | Learning Diffusion Priors from Observations by Expectation Maximization (Rozet et al., 2024)
    | https://arxiv.org/abs/2405.13712
"""

__all__ = [
    "MMPSDenoiser",
]

import torch

from functools import partial
from torch import Tensor
from typing import Callable

from ..denoise import Gaussian, GaussianDenoiser
from ..linalg.solve import cg, gmres
from ..noise import Schedule


class MMPSDenoiser(GaussianDenoiser):
    r"""Creates a MMPS denoiser module.

    Arguments:
        denoiser: A Gaussian denoiser.
        y: An observation :math:`y \sim \mathcal{N}(A(x), \Sigma_y)`, with shape :math:`(*, D)`.
        A: The forward operator :math:`x \mapsto A(x)`.
        var_y: The noise variance :math:`\Sigma_y`.
        tweedie_covariance: Whether to use the Tweedie covariance formula or not.
            If :py:`False`, use :math:`\Sigma_\phi(x_t)` instead.
        solver: The linear solver name (:py:`"cg"` or :py:`"gmres"`).
        iterations: The number of solver iterations.
    """

    def __init__(
        self,
        denoiser: GaussianDenoiser,
        y: Tensor,
        A: Callable[[Tensor], Tensor],
        var_y: Tensor,
        tweedie_covariance: bool = True,
        solver: str = "gmres",
        iterations: int = 1,
    ):
        super().__init__()

        self.denoiser = denoiser

        self.A = A

        self.register_buffer("y", torch.as_tensor(y))
        self.register_buffer("var_y", torch.as_tensor(var_y))

        self.tweedie_covariance = tweedie_covariance

        if solver == "cg":
            self.solve = partial(cg, iterations=iterations)
        elif solver == "gmres":
            self.solve = partial(gmres, iterations=iterations)
        else:
            raise ValueError(f"Unknown solver '{solver}'.")

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

        def A(v):
            return torch.func.jvp(self.A, (x_hat.detach(),), (v,))[1]

        def At(v):
            return torch.autograd.grad(y_hat, x_hat, v, retain_graph=True)[0]

        # fmt: off
        if self.tweedie_covariance:
            def cov_x(v):
                return gamma_t * torch.autograd.grad(x_hat, x_t, v, retain_graph=True)[0]
        else:
            def cov_x(v):
                return q.var * v
        # fmt: on

        def cov_y(v):
            return self.var_y * v + A(cov_x(At(v)))

        grad = self.y - y_hat
        grad = self.solve(A=cov_y, b=grad)
        grad = torch.autograd.grad(y_hat, x_t, grad)[0]

        return Gaussian(
            mean=x_hat + gamma_t * grad,
            var=q.var,
        )
