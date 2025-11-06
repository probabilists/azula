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

For most use cases, it is enough to estimate the mean :math:`\mathbb{E}[X \mid x_t]` of
the posterior :math:`p(X \mid x_t)`, in which case :math:`q_\phi(X \mid x_t)` should
have mean :math:`\mu_\phi(x_t) \approx \mathbb{E}[X \mid x_t]`.
"""

__all__ = [
    "Posterior",
    "DiracPosterior",
    "GaussianPosterior",
    "Denoiser",
    "PreconditionedDenoiser",
]

import abc
import math
import torch
import torch.nn as nn

from torch import Tensor
from typing import Callable, Self

from .linalg.covariance import Covariance, IsotropicCovariance
from .nn.utils import get_module_dtype
from .noise import Schedule


class Posterior(abc.ABC):
    r"""Abstract posterior :math:`q_\phi(X \mid x_t)`."""

    mean: Tensor


class DiracPosterior(Posterior):
    r"""Creates a Dirac delta posterior :math:`\delta(X - \mu)`.

    Arguments:
        mean: The mean :math:`\mu`, with shape :math:`(*)`.
    """

    mean: Tensor

    def __init__(self, mean: Tensor):
        self.mean = mean


class GaussianPosterior(Posterior):
    r"""Creates a Gaussian posterior :math:`\mathcal{N}(X \mid \mu, \sigma^2)`.

    Arguments:
        mean: The mean :math:`\mu`, with shape :math:`(*)`.
        var: The variance :math:`\sigma^2`, with shape :math:`(*)`.
    """

    mean: Tensor
    var: Tensor

    def __init__(self, mean: Tensor, var: Tensor):
        self.mean = mean
        self.var = var

    def log_prob(self, x: Tensor) -> Tensor:
        r"""
        Arguments:
            x: A tensor :math:`x`, with shape :math:`(*)`.

        Returns:
            The log-density :math:`\log \mathcal{N}(x \mid \mu, \sigma^2)`, with shape
            :math:`(*)`.
        """

        return -((x - self.mean) ** 2 / self.var + torch.log(self.var) + math.log(2 * math.pi)) / 2


class Denoiser(nn.Module):
    r"""Abstract denoiser module."""

    schedule: Schedule

    @abc.abstractmethod
    def forward(self, x_t: Tensor, t: Tensor, **kwargs) -> Posterior:
        r"""
        Arguments:
            x_t: A noisy tensor :math:`x_t`, with shape :math:`(B, *)`.
            t: The time :math:`t`, with shape :math:`()` or :math:`(B)`.
            kwargs: Optional keyword arguments.

        Returns:
            The posterior :math:`q_\phi(X \mid x_t)`.
        """

        pass


class GaussianDenoiser(Denoiser):
    r"""Creates an analytical Gaussian denoiser.

    Let :math:`X \sim \mathcal{N}(\mu_x, \Sigma_x)` and :math:`X_t \sim \mathcal{N}(X,
    \Sigma_t)`, then

    .. math:: X \mid X_t \sim \mathcal{N} \left( \bar{\mu}, \bar{\Sigma} \right)

    with

    .. math::
        \bar{\mu} & = \mu_x + \Sigma_x \left( \Sigma_x + \Sigma_t \right)^{-1} (X_t - \mu_x) \\
        \bar{\Sigma} & = \left( \Sigma_x^{-1} + \Sigma_t^{-1} \right)^{-1}

    References:
        | Bayesian Filtering and Smoothing (Särkkä, 2013)
        | http://www.cambridge.org/9781108926645

    Arguments:
        mean: The mean vector :math:`\mu_x`, with shape :math:`(N_1, ..., N_d)`.
        cov: The covariance matrix :math:`\Sigma_x`, with shape
            :math:`(N_1, N_1, ..., N_d, N_d)`.
    """

    def __init__(self, mean: Tensor, cov: Covariance, schedule: Schedule):
        super().__init__()

        self.mean = mean
        self.cov = cov
        self.schedule = schedule

    def _apply(self, fn: Callable, recurse: bool = True) -> Self:
        super()._apply(fn, recurse=recurse)

        self.mean = fn(self.mean)
        self.cov = fn(self.cov)

        return self

    def forward(self, x_t: Tensor, t: Tensor, **kwargs) -> DiracPosterior:
        r"""
        Arguments:
            x_t: A noisy tensor :math:`x_t`, with shape :math:`(B, *)`.
            t: The time :math:`t`, with shape :math:`()`.
            kwargs: Optional keyword arguments.

        Returns:
            The Dirac delta :math:`\delta(X - \bar{\mu})`.
        """

        alpha_t, sigma_t = self.schedule(t)

        cov_t = IsotropicCovariance(sigma_t**2)
        cov_inv = (self.cov + cov_t).inv

        mean = self.mean + self.cov(cov_inv(x_t / alpha_t - self.mean))

        return DiracPosterior(mean=mean)


class PreconditionedDenoiser(Denoiser):
    r"""Creates a Gaussian denoiser with EDM-style preconditioning.

    .. math::
        \mu_\phi(x_t) & = c_\mathrm{skip}(t) \, x_t +
            c_\mathrm{out}(t) \, b_\phi(c_\mathrm{in}(t) \, x_t, c_\mathrm{time}(t)) \\
        \sigma^2_\phi(x_t) & = \frac{\sigma_t^2}{\alpha_t^2 + \sigma_t^2}

    The preconditioning coefficients are generalized to take the scale :math:`\alpha_t`
    into account.

    .. math::
        c_\mathrm{in}(t) & = \frac{1}{\sqrt{\alpha_t^2 + \sigma_t^2}} \\
        c_\mathrm{out}(t) & = \frac{\sigma_t}{\sqrt{\alpha_t^2 + \sigma_t^2}} \\
        c_\mathrm{skip}(t) & = \frac{\alpha_t}{\alpha_t^2 + \sigma_t^2} \\
        c_\mathrm{time}(t) & = \log \frac{\sigma_t}{\alpha_t}

    References:
        | Elucidating the Design Space of Diffusion-Based Generative Models (Karras et al., 2022)
        | https://arxiv.org/abs/2206.00364

    Arguments:
        backbone: A noise/time conditional network :math:`b_\phi(x_t, t)`.
        schedule: A noise schedule.
    """

    def __init__(self, backbone: nn.Module, schedule: Schedule):
        super().__init__()

        self.backbone = backbone
        self.schedule = schedule

    def forward(self, x_t: Tensor, t: Tensor, **kwargs) -> GaussianPosterior:
        r"""
        Arguments:
            x_t: A noisy tensor :math:`x_t`, with shape :math:`(B, *)`.
            t: The time :math:`t`, with shape :math:`()` or :math:`(B)`.
            kwargs: Optional keyword arguments.

        Returns:
            The Gaussian :math:`\mathcal{N}(X \mid \mu_\phi(x_t), \Sigma_\phi(x_t))`.
        """

        alpha_t, sigma_t = self.schedule(t)

        while alpha_t.ndim < x_t.ndim:
            alpha_t, sigma_t = alpha_t[..., None], sigma_t[..., None]

        c_in = torch.rsqrt(alpha_t**2 + sigma_t**2)
        c_out = sigma_t * torch.rsqrt(alpha_t**2 + sigma_t**2)
        c_skip = alpha_t / (alpha_t**2 + sigma_t**2)
        c_time = torch.log(sigma_t / alpha_t).reshape_as(t)
        c_var = sigma_t**2 / (alpha_t**2 + sigma_t**2)

        dtype = get_module_dtype(self.backbone)

        output = self.backbone(
            (c_in * x_t).to(dtype),
            c_time.to(dtype),
            **kwargs,
        ).to(x_t)

        mean = c_skip * x_t + c_out * output
        var = c_var

        return GaussianPosterior(mean=mean, var=var)

    def loss(self, x: Tensor, t: Tensor, **kwargs) -> Tensor:
        r"""
        Arguments:
            x: A clean tensor :math:`x`, with shape :math:`(B, *)`.
            t: The time :math:`t`, with shape :math:`(B)`.
            kwargs: Optional keyword arguments.

        Returns:
            The weighted loss

            .. math:: \frac{1}{\sigma^2_\phi(x_t)} || \mu_\phi(x_t) - x ||^2

            where :math:`x_t \sim p(X_t \mid x)`, with shape :math:`(B, *)`.
        """

        alpha_t, sigma_t = self.schedule(t)

        while alpha_t.ndim < x.ndim:
            alpha_t, sigma_t = alpha_t[..., None], sigma_t[..., None]

        z = torch.randn_like(x)
        x_t = alpha_t * x + sigma_t * z

        q = self(x_t, t, **kwargs)

        return ((q.mean - x).square() / q.var.detach()).mean()
