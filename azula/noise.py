r"""Noise schedules.

A noise schedule is a mapping from a time :math:`t \in [0, 1]` to the signal scale
:math:`\alpha_t \in \mathbb{R}_+` and the noise scale :math:`\sigma_t \in \mathbb{R}_+`
in a perturbation kernel

.. math:: p(x_t \mid x) = \mathcal{N}(x_t \mid \alpha_t x_t, \sigma_t^2 I)

from a "clean" random variable :math:`x` to a "noisy" random variable :math:`x_t`. The
only constraint is for the signal-to-noise (SNR) ratio :math:`\frac{\alpha_t}{\sigma_t}`
to be (strictly) monotonically decreasing with respect to the time :math:`t`. Typically,
the initial signal scale :math:`\alpha_0` is set to 1 and the initial noise is small
enough (:math:`0 < \sigma_0 \ll 1`) that :math:`x_0` is almost equivalent to :math:`x`.

Note that the relation between :math:`x_s` and :math:`x_t` (:math:`0 \leq s \leq t`) is
not enforced by the noise schedule. For example,

.. math::
    z & \sim \mathcal{N}(0, I) \\
    x_s & = \alpha_s x + \sigma_s z \\
    x_t & = \alpha_t x + \sigma_t z

and

.. math::
    z_1, z_2 & \sim \mathcal{N}(0, I) \\
    x_s & = \alpha_s x + \sigma_s z_1 \\
    x_t & = \frac{\alpha_t}{\alpha_s} x_s + \sqrt{\sigma_t^2 - \frac{\alpha_t^2}{\alpha_s^2} \sigma_s^2} \, z_2

are both compatible with the perturbation kernel :math:`p(x_t \mid x)`.
"""

__all__ = [
    'Schedule',
    'VESchedule',
    'VPSchedule',
]

import abc
import torch
import torch.nn as nn

from torch import Tensor
from typing import Tuple


class Schedule(nn.Module, abc.ABC):
    r"""Abstract noise schedule."""

    @abc.abstractmethod
    def forward(self, t: Tensor) -> Tuple[Tensor, Tensor]:
        r"""
        Arguments:
            t: The time :math:`t`, with shape :math:`(*)`.

        Returns:
            The scales :math:`\alpha_t` and :math:`\sigma_t`, with shape :math:`(*)`.
        """

        pass


class VESchedule(Schedule):
    r"""Creates a variance exploding (VE) noise schedule.

    .. math::
        \alpha_t & = 1 \\
        \sigma_t & = \exp \big( (1 - t) \log \sigma_\min + t \log \sigma_\max \big)

    References:
        | Generative Modeling by Estimating Gradients of the Data Distribution (Song et al., 2019)
        | https://arxiv.org/pdf/1907.05600

        | Score-Based Generative Modeling through Stochastic Differential Equations (Song et al., 2021)
        | https://arxiv.org/abs/2011.13456

    Arguments:
        sigma_min: The initial noise scale :math:`\sigma_\min \in \mathbb{R}_+`.
        sigma_max: The final noise scale :math:`\sigma_\max \in \mathbb{R}_+`.
    """

    def __init__(self, sigma_min: float = 1e-3, sigma_max: float = 1e2):
        super().__init__()

        self.register_buffer('log_sigma_min', torch.log(torch.as_tensor(sigma_min)))
        self.register_buffer('log_sigma_max', torch.log(torch.as_tensor(sigma_max)))

    def alpha(self, t: Tensor) -> Tensor:
        return torch.ones_like(t)

    def sigma(self, t: Tensor) -> Tensor:
        return torch.exp(torch.lerp(self.log_sigma_min, self.log_sigma_max, t))

    def forward(self, t: Tensor) -> Tuple[Tensor, Tensor]:
        return self.alpha(t), self.sigma(t)


class VPSchedule(Schedule):
    r"""Creates a variance preserving (VP) noise schedule.

    .. math::
        \alpha_t & = \exp(-t^2 \log \alpha_\min) \\
        \sigma_t & = \sqrt{ 1 - \alpha_t^2 + \sigma_\min^2 }

    References:
        | Denoising Diffusion Probabilistic Models (Ho et al. 2020)
        | https://arxiv.org/abs/2006.11239

        | Score-Based Generative Modeling through Stochastic Differential Equations (Song et al., 2021)
        | https://arxiv.org/abs/2011.13456

    Arguments:
        alpha_min: The final signal scale :math:`\alpha_\min \in \mathbb{R}_+`.
        sigma_min: The initial noise scale :math:`\sigma_\min \in \mathbb{R}_+`.
    """

    def __init__(self, alpha_min: float = 1e-3, sigma_min: float = 1e-3):
        super().__init__()

        self.register_buffer('alpha_min', torch.as_tensor(alpha_min))
        self.register_buffer('sigma_min', torch.as_tensor(sigma_min))

    def alpha(self, t: Tensor) -> Tensor:
        return torch.exp(-torch.log(self.alpha_min) * t**2)

    def sigma(self, t: Tensor) -> Tensor:
        return torch.sqrt(1 - self.alpha(t) ** 2 + self.sigma_min**2)

    def forward(self, t: Tensor) -> Tuple[Tensor, Tensor]:
        return self.alpha(t), self.sigma(t)
