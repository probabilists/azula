r"""Reverse diffusion samplers.

For a distribution :math:`p(x)` over :math:`\mathbb{R}^D`, the perturbation kernel (see
:mod:`azula.noise`)

.. math:: p(x_t \mid x) = \mathcal{N}(x_t \mid \alpha_t x_t, \sigma_t^2 I)

defines a series of marginal distributions

.. math:: p(x_t) = \int p(x_t \mid x) \, p(x) \operatorname{d}\!x \, .

The goal of diffusion models is to generate samples from :math:`p(x_0)`. To do so,
reverse transition kernels :math:`q_\phi(x_s \mid x_t)` from :math:`x_t` to :math:`x_s`
(:math:`s < t`) are chosen. Then, starting from :math:`x_1 \sim p(x_1)`, :math:`T`
transitions :math:`q_\phi(x_{t_{i-1}} \mid x_{t_i})` where :math:`1 = t_T > \dots > t_1
> t_0 = 0` are simulated. If the kernels are consistent with the marginals
:math:`p(x_t)`, that is

.. math:: p(x_{t_{i-1}}) \approx
    \int q_\phi(x_{t_{i-1}} \mid x_{t_i}) \, p(x_{t_i}) \, \operatorname{d}\!x_{t_i} \, ,

for all :math:`i = 1, \dots, T`, the variables :math:`x_{t_i}` are approximately
distributed according to :math:`p(x_{t_i})`, including the last one :math:`x_{t_0} =
x_0`.
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
            The target vector :math:`x_s`, with shape :math:`(*, D)`.
        """

        pass


class DDPMSampler(Sampler):
    r"""Creates an DDPM sampler.

    .. math:: x_s = \alpha_s \mu_\phi(x_t, t)
        + \sigma_s \, \sqrt{1 - \tau} \, \frac{x_t - \alpha_t \mu_\phi(x_t, t)}{\sigma_t}
        + \sigma_s \, \sqrt{\tau} \, \epsilon

    where :math:`\epsilon \sim \mathcal{N}(0, I)` and

    .. math:: \tau = 1 - \frac{\alpha_t^2}{\alpha_s^2} \frac{\sigma_s^2}{\sigma_t^2} \, .

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

        tau = 1 - (alpha_t / alpha_s * sigma_s / sigma_t) ** 2
        eps = torch.randn_like(xt)

        mu = self.denoiser(xt, t)
        xs = alpha_s * mu
        xs = xs + sigma_s * torch.sqrt(1 - tau) / sigma_t * (xt - alpha_t * mu)
        xs = xs + sigma_s * torch.sqrt(tau) * eps

        return xs


class DDIMSampler(Sampler):
    r"""Creates a DDIM sampler.

    .. math:: x_s = \alpha_s \mu_\phi(x_t, t)
        + \sigma_s \, \sqrt{1 - \eta \tau} \, \frac{x_t - \alpha_t \mu_\phi(x_t, t)}{\sigma_t}
        + \sigma_s \, \sqrt{\eta \tau} \, \epsilon

    where :math:`\epsilon \sim \mathcal{N}(0, I)` and

    .. math:: \tau = 1 - \frac{\alpha_t^2}{\alpha_s^2} \frac{\sigma_s^2}{\sigma_t^2} \, .

    References:
        | Denoising Diffusion Implicit Models (Song et al., 2021)
        | https://arxiv.org/abs/2010.02502

    Arguments:
        denoiser: A denoiser model :math:`\mu_\phi(x_t, t)`.
        eta: The stochasticity hyperparameter :math:`\eta \in \mathbb{R}_+`.
            If :math:`\eta = 1`, :class:`DDIMSampler` is equivalent to :class:`DDPMSampler`.
        kwargs: Keyword arguments passed to :class:`Sampler`.
    """

    def __init__(self, denoiser: Denoiser, eta: float = 0.0, **kwargs):
        super().__init__(**kwargs)

        self.denoiser = denoiser

        self.register_buffer('eta', torch.as_tensor(eta))

    def step(self, xt: Tensor, t: Tensor, s: Tensor) -> Tensor:
        alpha_s, sigma_s = self.denoiser.schedule(s)
        alpha_t, sigma_t = self.denoiser.schedule(t)

        tau = 1 - (alpha_t / alpha_s * sigma_s / sigma_t) ** 2
        tau = torch.clip(self.eta * tau, min=0.0, max=1.0)

        eps = torch.randn_like(xt)

        mu = self.denoiser(xt, t)
        xs = alpha_s * mu
        xs = xs + sigma_s * torch.sqrt(1 - tau) / sigma_t * (xt - alpha_t * mu)
        xs = xs + sigma_s * torch.sqrt(tau) * eps

        return xs
