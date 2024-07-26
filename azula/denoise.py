r"""Denoisers, parametrizations and training objectives.

For a distribution :math:`p(x)` over :math:`\mathbb{R}^D` and a perturbation kernel

.. math:: p(x_t \mid x) = \mathcal{N}(x_t \mid \alpha_t x, \sigma_t^2 I) \, ,

the optimal denoiser is the mean :math:`\mathbb{E}[x \mid x_t]` of the denoising
posterior

.. math:: p(x \mid x_t) = \frac{p(x) \, p(x_t \mid x)}{p(x_t)} \, .

Typically, the optimal denoiser is unknown, but can be approximated by a neural network
:math:`\mu_\phi(x_t, t)` trained to minimize the denoising objective

.. math:: \arg\min_\phi
    \mathbb{E}_{\mathcal{U}(t | 0, 1)} \mathbb{E}_{p(x)} \mathbb{E}_{p(x_t \mid x)}
    \big[ \lambda_t \| \mu_\phi(x_t, t) - x \|^2 \big]

where :math:`\lambda_t \in \mathbb{R}_+` is a positive weight.
"""

__all__ = [
    'Denoiser',
    'DenoiserLoss',
    'EDMDenoiser',
]

import abc
import torch
import torch.nn as nn

from torch import Tensor

# isort: split
from .noise import Schedule


class Denoiser(nn.Module):
    r"""Abstract denoiser module.

    .. math:: \mu_\phi(x_t, t) \approx \mathbb{E}[x \mid x_t]

    Arguments:
        schedule: A noise schedule.
    """

    def __init__(self, schedule: Schedule):
        super().__init__()

        self.schedule = schedule

    @abc.abstractmethod
    def forward(self, xt: Tensor, t: Tensor, **kwargs) -> Tensor:
        r"""
        Arguments:
            xt: The noisy vector :math:`x_t`, with shape :math:`(*, D)`.
            sigma_t: The time :math:`t`, with shape :math:`(*)`.
            kwargs: Optional keyword arguments.

        Returns:
            The denoising estimate :math:`\mu_\phi(x_t, t)`, with shape :math:`(*, D)`.
        """

        pass


class DenoiserLoss(nn.Module):
    r"""Creates a module that calculates the denoising loss.

    Note:
        The weight :math:`\lambda_t` is set to the inverse of the variance of :math:`p(x
        \mid x_t)` when :math:`p(x) = \mathcal{N}(x \mid 0, 1)`, that is

        .. math:: \lambda_t = \frac{\alpha_t^2}{\sigma_t^2} + 1 \, .

    Arguments:
        denoiser: A denoiser :math:`\mu_\phi(x_t, t)`.
    """

    def __init__(self, denoiser: Denoiser):
        super().__init__()

        self.denoiser = denoiser

    def forward(self, x: Tensor, t: Tensor, **kwargs) -> Tensor:
        r"""
        Arguments:
            x: A batch of clean vectors :math:`x_i`, with shape :math:`(N, D)`.
            t: A batch of times :math:`t_i`, with shape :math:`(N)`.
            kwargs: Optional keyword arguments passed to :py:`self.denoiser`.

        Returns:
            The loss :math:`L`, with shape :math:`()`.

            .. math:: L = \frac{1}{N \times D} \sum_{i = 1}^{N}
                \lambda_{t_i} \| \mu_\phi(\alpha_{t_i} x_i + \sigma_{t_i} z_i, t_i) - x_i \|^2

            where :math:`z_i \sim \mathcal{N}(0, I)`.
        """

        alpha_t, sigma_t = self.denoiser.schedule(t)
        lmbda_t = (alpha_t / sigma_t) ** 2 + 1

        xt = alpha_t[..., None] * x + sigma_t[..., None] * torch.randn_like(x)

        error = torch.square(self.denoiser(xt, t, **kwargs) - x)
        loss = torch.mean(lmbda_t * torch.mean(error, dim=-1))

        return loss


class EDMDenoiser(Denoiser):
    r"""Creates a denoiser module with EDM-style preconditioning.

    .. math:: \mu_\phi(x_t, t) = c_s(t) \, x_t + c_o(t) \, b_\phi(c_i(t) \, x_t, c_n(t))

    The preconditioning coefficients are generalized to take the scale :math:`\alpha_t`
    into account.

    .. math::
        c_i(t) & = \frac{1}{\sqrt{\alpha_t^2 + \sigma_t^2}} \\
        c_o(t) & = \frac{\sigma_t}{\sqrt{\alpha_t^2 + \sigma_t^2}} \\
        c_s(t) & = \frac{\alpha_t}{\alpha_t^2 + \sigma_t^2} \\
        c_n(t) & = \log \frac{\sigma_t}{\alpha_t}

    References:
        | Elucidating the Design Space of Diffusion-Based Generative Models (Karras et al., 2022)
        | https://arxiv.org/abs/2206.00364

    Arguments:
        backbone: A noise conditional network :math:`b_\phi(x, \log \sigma)`.
        kwargs: Keyword arguments passed to :class:`Denoiser`.
    """

    def __init__(self, backbone: nn.Module, **kwargs):
        super().__init__(**kwargs)

        self.backbone = backbone

    def forward(self, xt: Tensor, t: Tensor, **kwargs) -> Tensor:
        alpha_t, sigma_t = self.schedule(t)

        c_in = 1 / torch.sqrt(alpha_t**2 + sigma_t**2)
        c_out = sigma_t / torch.sqrt(alpha_t**2 + sigma_t**2)
        c_skip = alpha_t / (alpha_t**2 + sigma_t**2)
        c_noise = torch.log(sigma_t / alpha_t)

        c_in, c_out, c_skip = c_in[..., None], c_out[..., None], c_skip[..., None]

        return c_skip * xt + c_out * self.backbone(c_in * xt, c_noise, **kwargs)
