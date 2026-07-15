r"""Predict-Project-Renoise (PPR) internals.

References:
    | Predict-Project-Renoise: Sampling Diffusion Models under Hard Constraints (Rochman-Sharabi et al., 2026)
    | https://arxiv.org/abs/2601.21033
"""

__all__ = ["PPRSampler"]

import torch

from collections.abc import Callable
from torch import Tensor
from torch.optim import Optimizer

from ..denoise import Denoiser
from ..sample import PCSampler


class PPRSampler(PCSampler):
    r"""Creates a PPR sampler.

    At each step, performs a predict-project-renoise (PPR) iteration: a predictor-corrector
    step is followed by :math:`M` rounds of projecting :math:`x_s` toward the
    constraint :math:`\mathcal{Y} = \{x : A(x) = y\}` and re-noising via the
    forward process :math:`p(X_t \mid X)`.

    Arguments:
        denoiser: A denoiser :math:`q_\phi(X \mid X_t)`.
        y: Constraint value :math:`y = A(x_0)`.
        A: The constraint operator :math:`x \mapsto A(x)`.
        porj_steps: Number of optimizer steps per projection.
        num_renoise: Number of project-then-renoise iterations :math:`M` per sampling step. If ``0``, there will be a projection but no re-noising.
        optimizer: A factory ``lambda p: Optimizer(p)`` called once per projection to
            build the optimizer. Defaults to :class:`torch.optim.Adam` with ``lr``.
        lr: Learning rate for the default optimizer.
        corrections: The number of corrector steps per predictor step. Defaults to ``0``
            unlike :class:`azula.sample.PCSampler` where the default is ``1``, since the
            projection already acts as a correction.
        kwargs: Keyword arguments passed to :class:`azula.sample.PCSampler`.
    """

    def __init__(
        self,
        denoiser: Denoiser,
        y: Tensor,
        A: Callable[[Tensor], Tensor],
        porj_steps: int = 1,
        num_renoise: int = 1,
        optimizer: Callable[[list[Tensor]], Optimizer] | None = None,
        lr: float = 1e-1,
        corrections: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(denoiser, corrections=corrections, **kwargs)
        self.y = y
        self.A = A
        self.porj_steps = porj_steps
        self.num_renoise = num_renoise
        self.optimizer = (
            optimizer if optimizer is not None else lambda p: torch.optim.Adam(p, lr=lr)
        )

    def _project(self, x_s: Tensor, s: Tensor, **kwargs) -> tuple[Tensor, Tensor]:
        with torch.enable_grad():
            x_s = x_s.detach().requires_grad_(True)
            opt = self.optimizer([x_s])

            def closure() -> Tensor:
                opt.zero_grad()
                loss = ((self.A(self.denoiser(x_s, s, **kwargs).mean) - self.y) ** 2).mean()
                loss.backward()
                return loss

            for _ in range(self.porj_steps):
                loss = opt.step(closure)
        return x_s.detach(), loss.detach()

    @torch.no_grad()
    def step(self, x_t: Tensor, t: Tensor, s: Tensor, **kwargs) -> Tensor:
        x_s = super().step(x_t, t, s, **kwargs)
        alpha_s, sigma_s = self.denoiser.schedule(s)

        if self.num_renoise == 0 and self.porj_steps > 0:
            x_s, _ = self._project(x_s, s, **kwargs)
        else:
            for _ in range(self.num_renoise):
                x_s, _ = self._project(x_s, s, **kwargs)
                x_s = alpha_s * self.denoiser(x_s, s, **kwargs).mean + sigma_s * torch.randn_like(
                    x_s
                )

        return x_s
