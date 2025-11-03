r"""Noise schedules.

A noise schedule is a mapping from a time :math:`t \in [0, 1]` to the signal scale
:math:`\alpha_t \in \mathbb{R}_+` and the noise scale :math:`\sigma_t \in \mathbb{R}_+`
in a perturbation kernel

.. math:: p(X_t \mid X) = \mathcal{N}(X_t \mid \alpha_t X_t, \sigma_t^2 I)

from a "clean" random variable :math:`X` to a "noisy" random variable :math:`X_t`. The
only constraint is for the signal-to-noise (SNR) ratio :math:`\frac{\alpha_t}{\sigma_t}`
to be monotonically decreasing with respect to the time :math:`t`. Typically, the
initial signal scale :math:`\alpha_0` is set to 1 and the initial noise is small enough
(:math:`0 < \sigma_0 \ll 1`) that :math:`X_0` is almost equivalent to :math:`X`.

Note that the relation between :math:`X_s` and :math:`X_t` (:math:`0 \leq s \leq t`) is
not enforced by the noise schedule. For example,

.. math::
    Z & \sim \mathcal{N}(0, I) \\
    X_s & = \alpha_s X + \sigma_s Z \\
    X_t & = \alpha_t X + \sigma_t Z

and

.. math::
    Z_1, Z_2 & \sim \mathcal{N}(0, I) \\
    X_s & = \alpha_s X + \sigma_s Z_1 \\
    X_t & = \frac{\alpha_t}{\alpha_s} X_s + \sqrt{\sigma_t^2 - \frac{\alpha_t^2}{\alpha_s^2} \sigma_s^2} \, Z_2

are both compatible with the perturbation kernel :math:`p(X_t \mid X)`.
"""

__all__ = [
    "Schedule",
    "VESchedule",
    "VPSchedule",
    "CosineSchedule",
    "RectifiedSchedule",
    "DecaySchedule",
]

import abc
import math
import torch

from torch import Tensor
from typing import Tuple


class Schedule(abc.ABC):
    r"""Abstract noise schedule."""

    @abc.abstractmethod
    def __call__(self, t: Tensor) -> Tuple[Tensor, Tensor]:
        r"""
        Arguments:
            t: The time :math:`t`, with shape :math:`(*)`.

        Returns:
            The signal and noise scales :math:`\alpha_t` and :math:`\sigma_t`, with
            shape :math:`(*)`.
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

    def __init__(self, sigma_min: float = 1e-3, sigma_max: float = 1e3):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, t: Tensor) -> Tuple[Tensor, Tensor]:
        return self.alpha(t), self.sigma(t)

    def alpha(self, t: Tensor) -> Tensor:
        return torch.ones_like(t)

    def sigma(self, t: Tensor) -> Tensor:
        return torch.exp((1 - t) * math.log(self.sigma_min) + t * math.log(self.sigma_max))


class VPSchedule(Schedule):
    r"""Creates a variance preserving (VP) noise schedule.

    .. math::
        \alpha_t & = \exp \big( t^2 \log \alpha_\min \big) \\
        \sigma_t & = \sqrt{ 1 - \alpha_t^2 + \sigma_\min^2}

    References:
        | Denoising Diffusion Probabilistic Models (Ho et al. 2020)
        | https://arxiv.org/abs/2006.11239

        | Score-Based Generative Modeling through Stochastic Differential Equations (Song et al., 2021)
        | https://arxiv.org/abs/2011.13456

    Arguments:
        alpha_min: The final signal scale :math:`\alpha_\min \in ]0,1[`.
        sigma_min: The initial noise scale :math:`\sigma_\min \in ]0,1[`.
    """

    def __init__(self, alpha_min: float = 1e-3, sigma_min: float = 1e-3):
        self.alpha_min = alpha_min
        self.sigma_min = sigma_min

    def __call__(self, t: Tensor) -> Tuple[Tensor, Tensor]:
        return self.alpha(t), self.sigma(t)

    def alpha(self, t: Tensor) -> Tensor:
        return torch.exp(math.log(self.alpha_min) * t**2)

    def sigma(self, t: Tensor) -> Tensor:
        return torch.sqrt(1 - self.alpha(t) ** 2 + self.sigma_min**2)


class CosineSchedule(Schedule):
    r"""Creates a cosine noise schedule.

    .. math::
        \alpha_t & = \cos \big( t \arccos \alpha_\min \big) \\
        \sigma_t & = \sqrt{ 1 - \alpha_t^2 + \sigma_\min^2}

    Arguments:
        alpha_min: The final signal scale :math:`\alpha_\min \in ]0,1[`.
        sigma_min: The initial noise scale :math:`\sigma_\min \in ]0,1[`.
    """

    def __init__(self, alpha_min: float = 1e-3, sigma_min: float = 1e-3):
        self.alpha_min = alpha_min
        self.sigma_min = sigma_min

    def __call__(self, t: Tensor) -> Tuple[Tensor, Tensor]:
        return self.alpha(t), self.sigma(t)

    def alpha(self, t: Tensor) -> Tensor:
        return torch.cos(math.acos(self.alpha_min) * t)

    def sigma(self, t: Tensor) -> Tensor:
        return torch.sqrt(1 - self.alpha(t) ** 2 + self.sigma_min**2)


class RectifiedSchedule(Schedule):
    r"""Creates a rectified noise schedule.

    .. math::
        \alpha_t & = t \, \alpha_\min + (1 - t) \\
        \sigma_t & = t + (1 - t) \, \sigma_\min

    References:
        | Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow (Liu et al. 2022)
        | https://arxiv.org/abs/2209.03003

        | Flow Matching for Generative Modeling (Lipman et al. 2023)
        | https://arxiv.org/abs/2210.02747

    Arguments:
        alpha_min: The final signal scale :math:`\alpha_\min \in ]0,1[`.
        sigma_min: The initial noise scale :math:`\sigma_\min \in ]0,1[`.
    """

    def __init__(self, alpha_min: float = 1e-3, sigma_min: float = 1e-3):
        self.alpha_min = alpha_min
        self.sigma_min = sigma_min

    def __call__(self, t: Tensor) -> Tuple[Tensor, Tensor]:
        return self.alpha(t), self.sigma(t)

    def alpha(self, t: Tensor) -> Tensor:
        return t * self.alpha_min + (1 - t)

    def sigma(self, t: Tensor) -> Tensor:
        return t + (1 - t) * self.sigma_min


class DecaySchedule(Schedule):
    r"""Creates an exponential decay schedule.

    .. math::
        \alpha_t & = \tau \, \alpha_\min + (1 - \tau) \\
        \sigma_t & = \tau + (1 - \tau) \, \sigma_\min

    where

    .. math:: \tau = \frac{1 - \gamma^t}{1 - \gamma}

    The smaller :math:`\gamma`, the more time is allocated to high signal-to-noise
    ratios. When :math:`\gamma` is close to 1, the schedule becomes equivalent to the
    :class:`RectifiedSchedule`.

    Arguments:
        alpha_min: The final signal scale :math:`\alpha_\min \in ]0,1[`.
        sigma_min: The initial noise scale :math:`\sigma_\min \in ]0,1[`.
        gamma: The decay factor :math:`\gamma \in ]0,1[`.
    """

    def __init__(self, alpha_min: float = 1e-3, sigma_min: float = 1e-3, gamma: float = 0.1):
        self.alpha_min = alpha_min
        self.sigma_min = sigma_min
        self.gamma = gamma

    def __call__(self, t: Tensor) -> Tuple[Tensor, Tensor]:
        return self.alpha(t), self.sigma(t)

    def tau(self, t: Tensor) -> Tensor:
        return (1 - self.gamma**t) / (1 - self.gamma)

    def alpha(self, t: Tensor) -> Tensor:
        t = self.tau(t)
        return t * self.alpha_min + (1 - t)

    def sigma(self, t: Tensor) -> Tensor:
        t = self.tau(t)
        return t + (1 - t) * self.sigma_min
