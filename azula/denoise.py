r"""Denoisers, parametrizations and training objectives.

For a distribution :math:`p(X)` over :math:`\mathbb{R}^D` and a perturbation kernel

.. math:: p(X_t \mid X) = \mathcal{N}(X_t \mid \alpha_t X, \sigma_t^2 I) \, ,

the goal of a denoiser is to predict :math:`X` given :math:`X_t`, that is to infer the
posterior distribution

.. math:: p(X \mid X_t) = \frac{p(X) \, p(X_t \mid X)}{p(X_t)} \, .

A denoiser is therefore an approximation :math:`q_\phi(X \mid X_t)` of the posterior
:math:`p(X \mid X_t)` and its objective is to minimize the Kullback-Leibler (KL)
divergence

.. math::
    & \arg \min_\phi \mathbb{E}_{x_t \,\sim\, p(X_t)}
        \big[ \mathrm{KL}( p(X \mid x_t) \parallel q_\phi(X \mid x_t) ) \big] \\
    = \, & \arg \min_\phi \mathbb{E}_{x, x_t \,\sim\, p(X, X_t)}
        \big[ -\log q_\phi(x \mid x_t) \big] \, .
"""

__all__ = [
    "Gaussian",
    "GaussianDenoiser",
    "PreconditionedDenoiser",
]

import abc
import math
import torch
import torch.nn as nn

from dataclasses import dataclass
from torch import Tensor

# isort: split
from .noise import Schedule


@dataclass
class Gaussian:
    r"""Creates a Gaussian distribution :math:`\mathcal{N}(\mu, \Sigma)`.

    Only diagonal covariances :math:`\Sigma = \operatorname{diag}(\sigma^2)` are
    currently supported.

    Arguments:
        mean: The mean :math:`\mu`, with shape :math:`(*, D)`.
        var: The variance :math:`\sigma^2`, with shape :math:`(*, D)`.
    """

    mean: Tensor
    var: Tensor

    def log_prob(self, x: Tensor) -> Tensor:
        r"""
        Arguments:
            x: A vector :math:`x`, with shape :math:`(*, D)`.

        Returns:
            The log-density :math:`\log \mathcal{N}(x \mid \mu, \Sigma)`, with shape
            :math:`(*)`.
        """

        log_p = (
            -((x - self.mean) ** 2 / self.var + torch.log(self.var) + math.log(2 * math.pi)) / 2
        )
        log_p = torch.sum(log_p, dim=-1)

        return log_p


class GaussianDenoiser(nn.Module):
    r"""Abstract Gaussian denoiser module.

    .. math:: q_\phi(X \mid X_t) = \mathcal{N}(X \mid \mu_\phi(X_t), \Sigma_\phi(X_t))

    The optimal Gaussian denoiser estimates the mean :math:`\mathbb{E}[X \mid X_t]` and
    covariance :math:`\mathbb{V}[X \mid X_t]` of the posterior :math:`p(X \mid X_t)`.
    """

    schedule: Schedule

    @abc.abstractmethod
    def forward(self, x_t: Tensor, t: Tensor, **kwargs) -> Gaussian:
        r"""
        Arguments:
            x_t: A noisy vector :math:`x_t`, with shape :math:`(*, D)`.
            t: The time :math:`t`, with shape :math:`(*)`.
            kwargs: Optional keyword arguments.

        Returns:
            The Gaussian :math:`\mathcal{N}(X \mid \mu_\phi(x_t), \Sigma_\phi(x_t))`.
        """

        pass

    def loss(self, x: Tensor, t: Tensor, **kwargs) -> Tensor:
        r"""
        Arguments:
            x: A clean vector :math:`x`, with shape :math:`(*, D)`.
            t: The time :math:`t`, with shape :math:`(*)`.
            kwargs: Optional keyword arguments.

        Returns:
            The negative log-likelihood

            .. math:: -\log \mathcal{N}(x \mid \mu_\phi(x_t), \Sigma_\phi(x_t))

            where :math:`x_t \sim p(X_t \mid x)`, with shape :math:`(*)`.
        """

        alpha_t, sigma_t = self.schedule(t)

        z = torch.randn_like(x)
        x_t = alpha_t * x + sigma_t * z

        q = self(x_t, t, **kwargs)

        return -q.log_prob(x)


class PreconditionedDenoiser(GaussianDenoiser):
    r"""Creates a Gaussian denoiser with EDM-style preconditioning.

    .. math::
        \mu_\phi(x_t) & = c_\mathrm{skip}(t) \, x_t +
            c_\mathrm{out}(t) \, b_\phi(c_\mathrm{in}(t) \, x_t, c_\mathrm{noise}(t)) \\
        \sigma^2_\phi(x_t) & = \frac{\sigma_t^2}{\alpha_t^2 + \sigma_t^2}

    The preconditioning coefficients are generalized to take the scale :math:`\alpha_t`
    into account.

    .. math::
        c_\mathrm{in}(t) & = \frac{1}{\sqrt{\alpha_t^2 + \sigma_t^2}} \\
        c_\mathrm{out}(t) & = \frac{\sigma_t}{\sqrt{\alpha_t^2 + \sigma_t^2}} \\
        c_\mathrm{skip}(t) & = \frac{\alpha_t}{\alpha_t^2 + \sigma_t^2} \\
        c_\mathrm{noise}(t) & = \log \frac{\sigma_t}{\alpha_t}

    References:
        | Elucidating the Design Space of Diffusion-Based Generative Models (Karras et al., 2022)
        | https://arxiv.org/abs/2206.00364

    Arguments:
        backbone: A noise conditional network :math:`b_\phi(x, \log \sigma)`.
        schedule: A noise schedule.
    """

    def __init__(self, backbone: nn.Module, schedule: Schedule):
        super().__init__()

        self.backbone = backbone
        self.schedule = schedule

    def forward(self, x_t: Tensor, t: Tensor, **kwargs) -> Gaussian:
        alpha_t, sigma_t = self.schedule(t)

        c_in = 1 / torch.sqrt(alpha_t**2 + sigma_t**2)
        c_out = sigma_t / torch.sqrt(alpha_t**2 + sigma_t**2)
        c_skip = alpha_t / (alpha_t**2 + sigma_t**2)
        c_noise = torch.log(sigma_t / alpha_t).squeeze(dim=-1)

        mean = c_skip * x_t + c_out * self.backbone(c_in * x_t, c_noise, **kwargs)
        var = sigma_t**2 / (alpha_t**2 + sigma_t**2)

        return Gaussian(mean=mean, var=var)
