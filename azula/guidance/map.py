r"""Maximum A Posteriori (MAP) sampling internals."""

__all__ = [
    "MAPDenoiser",
]

import torch

from collections.abc import Callable
from functools import partial
from torch import Tensor

from ..denoise import Denoiser, DiracPosterior
from ..linalg.covariance import Covariance, DiagonalCovariance
from ..linalg.solve import cg, gmres
from ..noise import Schedule


class MAPDenoiser(Denoiser):
    r"""Creates a MAP denoiser module.

    Arguments:
        denoiser: A denoiser :math:`q_\phi(X \mid X_t)`.
        y: An observation :math:`y \sim \mathcal{N}(A(x), \Sigma_y)`, with shape :math:`(*, D)`.
        A: The forward operator :math:`x \mapsto A(x)`.
        cov_y: The noise covariance :math:`\Sigma_y`. If `cov_y` is a tensor, it is
            assumed to be the variance :math:`\sigma_y^2` and :math:`\Sigma_y =
            \mathrm{diag}(\sigma_y^2)`.
        solver: The linear solver name (:py:`"cg"` or :py:`"gmres"`).
        iterations: The number of solver iterations.
    """

    def __init__(
        self,
        denoiser: Denoiser,
        y: Tensor,
        A: Callable[[Tensor], Tensor],
        cov_y: Tensor | Covariance,
        solver: str = "gmres",
        iterations: int = 1,
    ) -> None:
        super().__init__()

        self.denoiser = denoiser

        self.y = y
        self.A = A

        if isinstance(cov_y, Covariance):
            self.cov_y = cov_y
        else:
            self.cov_y = DiagonalCovariance(cov_y)

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
        gamma_t = sigma_t**2 / alpha_t

        with torch.enable_grad():
            x_t = x_t.detach().requires_grad_()
            q = self.denoiser(x_t, t, **kwargs)
            x_hat = q.mean

        def cov_x(v: Tensor) -> Tensor:
            return gamma_t * torch.autograd.grad(x_hat, x_t, v, retain_graph=True)[0]

        x = torch.nn.Parameter(x_hat.detach(), requires_grad=True)

        @torch.no_grad()
        def closure() -> torch.Tensor:
            delta_x = x_hat - x
            grad_x = self.solve(A=cov_x, b=delta_x)
            loss_x = torch.sum(delta_x * grad_x)

            with torch.enable_grad():
                delta_y = self.y - self.A(x)

            grad_y = self.cov_y.inv(delta_y)
            loss_y = torch.sum(delta_y * grad_y)

            x.grad = grad_x - torch.autograd.grad(delta_y, x, grad_y)

            return loss_x + loss_y

        torch.optim.LBFGS([x], lr=1e-2, max_iter=16).step(closure)

        return DiracPosterior(mean=x.detach())
