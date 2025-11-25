r"""Diffusion Plug-and-Play Image Restoration (DiffPIR) internals.

References:
    | Denoising Diffusion Models for Plug-and-Play Image Restoration (Zhu et al., 2023)
    | https://arxiv.org/abs/2305.08995
"""

__all__ = [
    "DiffPIRDenoiser",
]

import torch

from functools import partial
from torch import Tensor
from typing import Callable, Union

from ..denoise import Denoiser, DiracPosterior
from ..linalg.solve import cg, gmres
from ..noise import Schedule


class DiffPIRDenoiser(Denoiser):
    r"""Creates a DiffPIR denoiser module.

    Arguments:
        denoiser: A denoiser :math:`q_\phi(X \mid X_t)`.
        y: An observation :math:`y \sim \mathcal{N}(A x, \Sigma_y)`, with shape :math:`(*, D)`.
        A: The forward operator :math:`x \mapsto A x`.
        var_y: The noise variance :math:`\Sigma_y`.
        lmbda: The regularization strength :math:`\lambda \in \mathbb{R}_+`.
        solver: The linear solver name (:py:`"cg"` or :py:`"gmres"`).
        iterations: The number of solver iterations.
    """

    def __init__(
        self,
        denoiser: Denoiser,
        y: Tensor,
        A: Callable[[Tensor], Tensor],
        var_y: Union[float, Tensor],
        lmbda: float = 10.0,
        solver: str = "gmres",
        iterations: int = 1,
    ):
        super().__init__()

        self.denoiser = denoiser

        self.y = y
        self.A = A
        self.var_y = var_y
        self.lmbda = lmbda

        if solver == "cg":
            self.solve = partial(cg, iterations=iterations)
        elif solver == "gmres":
            self.solve = partial(gmres, iterations=iterations)
        else:
            raise ValueError(f"Unknown solver '{solver}'.")

    @property
    def schedule(self) -> Schedule:
        return self.denoiser.schedule

    @torch.no_grad()
    def forward(self, x_t: Tensor, t: Tensor, **kwargs) -> DiracPosterior:
        alpha_t, sigma_t = self.schedule(t)
        rho_t = (sigma_t / alpha_t) ** 2

        q = self.denoiser(x_t, t, **kwargs)

        with torch.enable_grad():
            x_hat = q.mean.detach().requires_grad_()
            y_hat = self.A(x_hat)

        def At(v):
            return torch.autograd.grad(y_hat, x_hat, v, retain_graph=True)[0]

        def AtA_I(v):
            return At(self.A(v) / self.var_y) + self.lmbda * v / rho_t

        grad = (self.y - y_hat) / self.var_y
        grad = At(grad)
        grad = self.solve(A=AtA_I, b=grad)

        return DiracPosterior(mean=x_hat + grad)
