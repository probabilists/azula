r"""Moment Matching Posterior Sampling (MMPS) internals.

References:
    | Learning Diffusion Priors from Observations by Expectation Maximization (Rozet et al., 2024)
    | https://arxiv.org/abs/2405.13712
"""

__all__ = [
    "MMPSDenoiser",
    "YeetSampler",
]

import torch

from collections.abc import Callable
from functools import partial
from torch import Tensor

from ..denoise import Denoiser, DiracPosterior
from ..linalg.covariance import Covariance, DiagonalCovariance
from ..linalg.solve import cg, gmres
from ..noise import Schedule
from ..sample import DDIMSampler


class MMPSDenoiser(Denoiser):
    r"""Creates a MMPS denoiser module.

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
            y_hat = self.A(x_hat)

        def A(v: Tensor) -> Tensor:
            return torch.func.jvp(self.A, (x_hat.detach(),), (v,))[1]

        def At(v: Tensor) -> Tensor:
            return torch.autograd.grad(y_hat, x_hat, v, retain_graph=True)[0]

        def cov_x(v: Tensor) -> Tensor:
            return gamma_t * torch.autograd.grad(x_hat, x_t, v, retain_graph=True)[0]

        def cov_y(v: Tensor) -> Tensor:
            return self.cov_y(v) + A(cov_x(At(v)))

        grad = self.y - y_hat
        grad = self.solve(A=cov_y, b=grad)
        grad = gamma_t * torch.autograd.grad(y_hat, x_t, grad)[0]

        return DiracPosterior(mean=x_hat + grad)


class YeetSampler(DDIMSampler):
    r"""Creates a YEET sampler.

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
        **kwargs,
    ) -> None:
        super().__init__(denoiser, **kwargs)

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

        # MMPS
        gamma_t = sigma_t**2 / alpha_t

        with torch.enable_grad():
            y_hat = self.A(x_hat)

        def A(v: Tensor) -> Tensor:
            return torch.func.jvp(self.A, (x_hat.detach(),), (v,))[1]

        def At(v: Tensor) -> Tensor:
            return torch.autograd.grad(y_hat, x_hat, v, retain_graph=True)[0]

        def cov_x(v: Tensor) -> Tensor:
            return gamma_t * torch.autograd.grad(x_hat, x_t, v, retain_graph=True)[0]

        def cov_y(v: Tensor) -> Tensor:
            return self.cov_y(v) + A(cov_x(At(v)))

        grad = self.y - y_hat
        grad = self.solve(A=cov_y, b=grad)
        grad = gamma_t * torch.autograd.grad(y_hat, x_t, grad)[0]

        return x_s + alpha_s * grad
