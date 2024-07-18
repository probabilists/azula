r"""Reverse diffusion samplers.

For a distribution :math:`p(x)` over :math:`\mathbb{R}^D`, the perturbation kernel (see
:mod:`azula.noise`)

.. math:: p(x_t \mid x) = \mathcal{N}(x_t \mid \alpha_t x_t, \sigma_t^2 I)

defines a series of marginal distributions

.. math:: p(x_t) = \int p(x_t \mid x) \, p(x) \operatorname{d}\!x \, .

The goal of diffusion models is to generate samples from :math:`p(x_0)`. To do so,
reverse transition kernels :math:`p(x_s \mid x_t)` from :math:`x_t` to :math:`x_s`
(:math:`s < t`) consistent with the marginals :math:`p(x_t)` are chosen. Then, starting
from :math:`x_1 \sim p(x_1)`, :math:`x_t` is denoised by simulating :math:`T`
transitions :math:`p(x_{t_{i-1}} \mid x_{t_i})` such that :math:`1 = t_T > ... > t_1 >
t_0 = 0`.
"""

__all__ = [
    'Sampler',
    'DDIMSampler',
    'DDPMSampler',
]

import abc
import torch
import torch.nn as nn

from torch import Tensor

# isort: split
from .denoise import Denoiser


class Sampler(nn.Module, abc.ABC):
    r"""Abstract reverse diffusion sampler.

    Arguments:
        steps: The number of discretization steps :math:`T`. By default, the step size
            :math:`t - s` is constant.
    """

    def __init__(self, steps: int):
        super().__init__()

        self.register_buffer('timesteps', torch.linspace(1, 0, steps + 1))

    def forward(self, x1: Tensor) -> Tensor:
        r"""Simulates the reverse process from :math:`t = 1` to :math:`0`.

        Arguments:
            x1: The noisy vector :math:`x_1`, with shape :math:`(*, D)`.

        Returns:
            The clean vector :math:`x_0`, with shape :math:`(*, D)`.
        """

        xt = x1

        for t, s in self.timesteps.unfold(0, 2, 1):
            xt = self.step(xt, t, s)

        x0 = xt

        return x0

    @abc.abstractmethod
    def step(self, xt: Tensor, t: Tensor, s: Tensor) -> Tensor:
        r"""Simulates the reverse process from :math:`t` to :math:`s \leq t`.

        Arguments:
            xt: The current vector :math:`x_t`, with shape :math:`(*, D)`.
            t: The current time :math:`t`, with shape :math:`(*)`.
            s: The target time :math:`s`, with shape :math:`(*)`.

        Returns:
            The target vector :math:`x_0`, with shape :math:`(*, D)`.
        """

        pass


class DDIMSampler(Sampler):
    r"""Creates a DDIM sampler.

    .. math:: \frac{x_s}{\alpha_s} = \tau \frac{x_t}{\alpha_t} + (1 - \tau) \mu_\phi(x_t, t)

    where :math:`\tau = \frac{\alpha_t}{\alpha_s} \frac{\sigma_s}{\sigma_t}`.

    References:
        | Denoising Diffusion Implicit Models (Song et al., 2021)
        | https://arxiv.org/abs/2010.02502

    Arguments:
        denoiser: A denoiser model :math:`\mu_\phi(x_t, t)`.
        kwargs: Keyword arguments passed to :class:`Sampler`.
    """

    def __init__(self, denoiser: Denoiser, **kwargs):
        super().__init__(**kwargs)

        self.denoiser = denoiser

    def step(self, xt: Tensor, t: Tensor, s: Tensor) -> Tensor:
        alpha_s, sigma_s = self.denoiser.schedule(s)
        alpha_t, sigma_t = self.denoiser.schedule(t)

        tau = (alpha_t / alpha_s * sigma_s / sigma_t) ** 2

        mu = self.denoiser(xt, t)
        xs = alpha_s * (tau * xt / alpha_t + (1 - tau) * mu)

        return xs


class DDPMSampler(Sampler):
    r"""Creates an DDPM sampler.

    .. math:: \frac{x_s}{\alpha_s} =
        \tau \frac{x_t}{\alpha_t}
        + (1 - \tau) \mu_\phi(x_t, t)
        + \frac{\sigma_s}{\alpha_s} \sqrt{1 - \tau} z

    where :math:`\tau = \frac{\alpha_t^2}{\alpha_s^2} \frac{\sigma_s^2}{\sigma_t^2}` and
    :math:`z \sim \mathcal{N}(0, I)`.

    References:
        | Denoising Diffusion Probabilistic Models (Ho et al., 2020)
        | https://arxiv.org/abs/2006.11239

    Arguments:
        denoiser: A denoiser :math:`\mu_\phi(x_t, t)`.
        kwargs: Keyword arguments passed to :class:`Sampler`.
    """

    def __init__(self, denoiser: Denoiser, **kwargs):
        super().__init__(**kwargs)

        self.denoiser = denoiser

    def step(self, xt: Tensor, t: Tensor, s: Tensor) -> Tensor:
        alpha_s, sigma_s = self.denoiser.schedule(s)
        alpha_t, sigma_t = self.denoiser.schedule(t)

        tau = (alpha_t / alpha_s * sigma_s / sigma_t) ** 2

        mu = self.denoiser(xt, t)
        xs = alpha_s * (tau * xt / alpha_t + (1 - tau) * mu)
        xs = xs + sigma_s * torch.sqrt(1 - tau) * torch.randn_like(xs)

        return xs
