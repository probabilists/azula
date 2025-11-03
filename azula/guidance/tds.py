r"""Twisted Posterior Sampling (TDS) internals.

References:
    | Practical and Asymptotically Exact Conditional Sampling in Diffusion Models (Wu et al., 2023)
    | https://arxiv.org/abs/2306.17775
"""

__all__ = [
    "TDSSampler",
]

import torch

from einops import reduce
from torch import Tensor
from torch.distributions import Normal
from typing import Callable

from ..denoise import Denoiser
from ..sample import Sampler


class TDSSampler(Sampler):
    r"""Creates a TDS sampler.

    Arguments:
        denoiser: A denoiser :math:`q_\phi(X \mid X_t)`.
        twist: A twisting function $\log p(y | \hat{x}, t)$.
        kwargs: Keyword arguments passed to :class:`azula.sample.Sampler`.
    """

    def __init__(
        self,
        denoiser: Denoiser,
        twist: Callable[[Tensor], Tensor],
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.denoiser = denoiser
        self.twist = twist

    @torch.no_grad()
    def __call__(self, x: Tensor, **kwargs) -> Tensor:
        r"""Simulates the reverse process from :math:`t_T` to :math:`t_0`.

        Arguments:
            x: A set of :math:`K` noisy tensors :math:`x_{t_T}`, with shape :math:`(K, *)`.
            kwargs: Optional keyword arguments.

        Returns:
            The clean(er) tensors :math:`x_{t_0}`, with shape :math:`(K, *)`.
        """

        return super().__call__(x, carry={}, **kwargs)

    def step(self, x_t: Tensor, t: Tensor, s: Tensor, carry: dict, **kwargs) -> Tensor:
        alpha_s, sigma_s = self.denoiser.schedule(s)
        alpha_t, sigma_t = self.denoiser.schedule(t)

        with torch.enable_grad():
            x_t = x_t.detach().requires_grad_()
            x_hat = self.denoiser(x_t, t, **kwargs).mean

            log_p_y = self.twist(x_hat, sigma_t / alpha_t)
            score_y = torch.autograd.grad(log_p_y.sum(), x_t)[0]

        # Resample
        log_p_y = reduce(log_p_y, "K ... -> K", "sum")

        if "log_w" in carry:
            log_w = log_p_y + carry["log_w"]
        else:
            log_w = log_p_y

        w = torch.softmax(log_w, dim=0)
        k = torch.multinomial(w, len(w), replacement=True)

        x_t, x_hat, log_p_y, score_y = x_t[k], x_hat[k], log_p_y[k], score_y[k]

        # Proposal
        def ddpm(x_t, x):
            eps = (x_t - alpha_t * x) / sigma_t
            tau = (alpha_t / alpha_s * sigma_s / sigma_t) ** 2

            return Normal(
                loc=alpha_s * x + sigma_s * torch.sqrt(tau) * eps,
                scale=sigma_s * torch.sqrt(1 - tau),
                validate_args=False,
            )

        q_s = ddpm(x_t, x_hat)
        q_s_y = ddpm(x_t, x_hat + sigma_t**2 / alpha_t * score_y)

        x_s = q_s_y.sample()

        # Reweight
        log_q_xs = reduce(q_s.log_prob(x_s), "K ... -> K", "sum")
        log_q_xs_y = reduce(q_s_y.log_prob(x_s), "K ... -> K", "sum")

        carry["log_w"] = log_q_xs - log_q_xs_y - log_p_y

        return x_s
