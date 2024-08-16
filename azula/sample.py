r"""Reverse diffusion samplers.

For a distribution :math:`p(X)` over :math:`\mathbb{R}^D`, the perturbation kernel (see
:mod:`azula.noise`)

.. math:: p(X_t \mid X) = \mathcal{N}(X_t \mid \alpha_t X_t, \sigma_t^2 I)

defines a series of marginal distributions

.. math:: p(X_t) = \int p(X_t \mid x) \, p(x) \operatorname{d}\!x \, .

The goal of diffusion models is to generate samples from :math:`p(X_0)`. To this end,
reverse transition kernels :math:`q(X_s \mid X_t)` from :math:`t` to :math:`s < t` are
chosen. Then, starting from :math:`x_1 \sim p(X_1)`, :math:`T` transitions
:math:`x_{t_{i-1}} \sim q(X_{t_{i-1}} \mid x_{t_i})` are simulated from :math:`t_T = 1` to
:math:`t_0 = 0`. If the kernels are consistent with the marginals :math:`p(X_t)`, that
is if

.. math:: p(X_{t_{i-1}}) \approx
    \int q(X_{t_{i-1}} \mid x_{t_i}) \, p(x_{t_i}) \, \operatorname{d}\!x_{t_i} \, ,

for all :math:`i = 1, \dots, T`, the vectors :math:`x_{t_i}` are distributed according
to :math:`p(X_{t_i})`, including the last one :math:`x_{t_0} = x_0`.
"""

__all__ = [
    "Sampler",
    "DDPMSampler",
    "DDIMSampler",
]

import abc
import torch
import torch.nn as nn

from torch import Tensor

# isort: split
from .denoise import GaussianDenoiser


class Sampler(nn.Module, abc.ABC):
    r"""Abstract reverse diffusion sampler.

    Arguments:
        steps: The number of discretization steps :math:`T`. By default, the step size
            :math:`t - s` is constant.
    """

    def __init__(self, steps: int):
        super().__init__()

        self.register_buffer("timesteps", torch.linspace(1, 0, steps + 1))

    @torch.no_grad()
    def forward(self, x_1: Tensor, **kwargs) -> Tensor:
        r"""Simulates the reverse process from :math:`t = 1` to :math:`0`.

        Arguments:
            x_1: The noisy vector :math:`x_1`, with shape :math:`(*, D)`.
            kwargs: Optional keyword arguments.

        Returns:
            The clean vector :math:`x_0`, with shape :math:`(*, D)`.
        """

        x_t = x_1

        for t, s in self.timesteps.unfold(0, 2, 1):
            x_s = self.step(x_t, t, s, **kwargs)
            x_t = x_s

        x_0 = x_t

        return x_0

    @abc.abstractmethod
    def step(self, x_t: Tensor, t: Tensor, s: Tensor, **kwargs) -> Tensor:
        r"""Simulates the reverse process from :math:`t` to :math:`s \leq t`.

        Arguments:
            x_t: The current vector :math:`x_t`, with shape :math:`(*, D)`.
            t: The current time :math:`t`, with shape :math:`(*)`.
            s: The target time :math:`s`, with shape :math:`(*)`.
            kwargs: Optional keyword arguments.

        Returns:
            The new vector :math:`x_s \sim q(X_s \mid x_t)`, with shape :math:`(*, D)`.
        """

        pass


class DDPMSampler(Sampler):
    r"""Creates an DDPM sampler.

    .. math:: x_s = \alpha_s \mu_\phi(x_t)
        + \sigma_s \, \sqrt{1 - \tau} \, \frac{x_t - \alpha_t \mu_\phi(x_t)}{\sigma_t}
        + \sigma_s \, \sqrt{\tau} \, \epsilon

    where :math:`\epsilon \sim \mathcal{N}(0, I)` and

    .. math:: \tau = 1 - \frac{\alpha_t^2}{\alpha_s^2} \frac{\sigma_s^2}{\sigma_t^2} \, .

    References:
        | Denoising Diffusion Probabilistic Models (Ho et al., 2020)
        | https://arxiv.org/abs/2006.11239

    Arguments:
        denoiser: A Gaussian denoiser.
        kwargs: Keyword arguments passed to :class:`Sampler`.
    """

    def __init__(self, denoiser: GaussianDenoiser, **kwargs):
        super().__init__(**kwargs)

        self.denoiser = denoiser

    def step(self, x_t: Tensor, t: Tensor, s: Tensor, **kwargs) -> Tensor:
        alpha_s, sigma_s = self.denoiser.schedule(s)
        alpha_t, sigma_t = self.denoiser.schedule(t)

        tau = 1 - (alpha_t / alpha_s * sigma_s / sigma_t) ** 2
        eps = torch.randn_like(x_t)

        x_hat = self.denoiser(x_t, t, **kwargs).mean
        x_s = alpha_s * x_hat
        x_s = x_s + sigma_s * torch.sqrt(1 - tau) / sigma_t * (x_t - alpha_t * x_hat)
        x_s = x_s + sigma_s * torch.sqrt(tau) * eps

        return x_s


class DDIMSampler(Sampler):
    r"""Creates a DDIM sampler.

    .. math:: x_s = \alpha_s \mu_\phi(x_t)
        + \sigma_s \, \sqrt{1 - \eta \, \tau} \, \frac{x_t - \alpha_t \mu_\phi(x_t)}{\sigma_t}
        + \sigma_s \, \sqrt{\eta \, \tau} \, \epsilon

    where :math:`\epsilon \sim \mathcal{N}(0, I)` and

    .. math:: \tau = 1 - \frac{\alpha_t^2}{\alpha_s^2} \frac{\sigma_s^2}{\sigma_t^2} \, .

    References:
        | Denoising Diffusion Implicit Models (Song et al., 2021)
        | https://arxiv.org/abs/2010.02502

    Arguments:
        denoiser: A Gaussian denoiser.
        eta: The stochasticity hyperparameter :math:`\eta \in \mathbb{R}_+`.
            If :math:`\eta = 1`, :class:`DDIMSampler` is equivalent to :class:`DDPMSampler`.
        kwargs: Keyword arguments passed to :class:`Sampler`.
    """

    def __init__(self, denoiser: GaussianDenoiser, eta: float = 0.0, **kwargs):
        super().__init__(**kwargs)

        self.denoiser = denoiser

        self.register_buffer("eta", torch.as_tensor(eta))

    def step(self, x_t: Tensor, t: Tensor, s: Tensor, **kwargs) -> Tensor:
        alpha_s, sigma_s = self.denoiser.schedule(s)
        alpha_t, sigma_t = self.denoiser.schedule(t)

        tau = 1 - (alpha_t / alpha_s * sigma_s / sigma_t) ** 2
        tau = torch.clip(self.eta * tau, min=0, max=1)

        eps = torch.randn_like(x_t)

        x_hat = self.denoiser(x_t, t, **kwargs).mean
        x_s = alpha_s * x_hat
        x_s = x_s + sigma_s * torch.sqrt(1 - tau) / sigma_t * (x_t - alpha_t * x_hat)
        x_s = x_s + sigma_s * torch.sqrt(tau) * eps

        return x_s
