r"""Elucidated diffusion model (EDM) plugin.

This plugin depends on the `torch_utils` and `training` modules in the `NVlabs/edm
<https://github.com/NVlabs/edm>`_ repository. To use it, clone the repository to your
machine

.. code-block:: console

    git clone https://github.com/NVlabs/edm

and add it to your Python path before importing the plugin.

.. code-block:: python

    import sys; sys.path.append("path/to/edm")
    ...
    from azula.plugins import edm

References:
    | Elucidating the Design Space of Diffusion-Based Generative Models (Karras et al., 2022)
    | https://arxiv.org/abs/2206.00364
"""

__all__ = [
    "ElucidatedSchedule",
    "ElucidatedDenoiser",
    "load_model",
]

import pickle
import torch
import torch.nn as nn

from torch import Tensor
from typing import Optional, Tuple

from azula.denoise import Gaussian, GaussianDenoiser
from azula.hub import download
from azula.noise import Schedule

from ..utils import load_cards


class ElucidatedSchedule(Schedule):
    r"""Creates an elucidated noise schedule.

    .. math::
        \alpha_t & = 1 \\
        \sigma_t & = \left( (1 - t) \, {\sigma_\min}^\frac{1}{\rho}
            + t \, {\sigma_\max}^\frac{1}{\rho} \right)^\rho

    Arguments:
        sigma_min: The initial noise scale :math:`\sigma_\min \in \mathbb{R}_+`.
        sigma_max: The final noise scale :math:`\sigma_\max \in \mathbb{R}_+`.
        rho: A hyper-parameter :math:`\rho \in \mathbb{R}_+`.
    """

    def __init__(self, sigma_min: float = 0.002, sigma_max: float = 80.0, rho: float = 7.0):
        super().__init__()

        self.register_buffer("sigma_min", torch.as_tensor(sigma_min))
        self.register_buffer("sigma_max", torch.as_tensor(sigma_max))
        self.register_buffer("rho", torch.as_tensor(rho))

    def alpha(self, t: Tensor) -> Tensor:
        return torch.ones_like(t)

    def sigma(self, t: Tensor) -> Tensor:
        lower = self.sigma_min ** (1 / self.rho)
        upper = self.sigma_max ** (1 / self.rho)
        return torch.lerp(lower, upper, t) ** self.rho

    def forward(self, t: Tensor) -> Tuple[Tensor, Tensor]:
        return self.alpha(t), self.sigma(t)


class ElucidatedDenoiser(GaussianDenoiser):
    r"""Creates an elucidated denoiser.

    Arguments:
        backbone: A noise conditional network.
        schedule: A noise schedule. If :py:`None`, use :class:`ElucidatedSchedule` instead.
    """

    def __init__(self, backbone: nn.Module, schedule: Optional[Schedule] = None):
        super().__init__()

        self.backbone = backbone

        if schedule is None:
            self.schedule = ElucidatedSchedule()
        else:
            self.schedule = schedule

    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        label: Optional[Tensor] = None,
        **kwargs,
    ) -> Gaussian:
        r"""
        Arguments:
            x_t: A noisy tensor :math:`x_t`, with shape :math:`(B, 3, H, W)`.
            t: The time :math:`t`, with shape :math:`()` or :math:`(B)`.
            label: The class label :math:`c` as a one-hot vector, with shape :math:`(*, 1000)`.
            kwargs: Optional keyword arguments.

        Returns:
            The Gaussian :math:`\mathcal{N}(X \mid \mu_\phi(x_t \mid c), \Sigma_\phi(x_t \mid c))`.
        """

        alpha_t, sigma_t = self.schedule(t)

        while alpha_t.ndim < x_t.ndim:
            alpha_t, sigma_t = alpha_t[..., None], sigma_t[..., None]

        c_in = 1 / alpha_t
        c_time = (sigma_t / alpha_t).reshape_as(t)
        c_var = sigma_t**2 / (alpha_t**2 + sigma_t**2)

        mean = self.backbone(c_in * x_t, c_time, class_labels=label, **kwargs)
        var = c_var

        return Gaussian(mean=mean, var=var)


def load_model(name: str) -> GaussianDenoiser:
    r"""Loads a pre-trained EDM denoiser.

    Arguments:
        name: The pre-trained model name.

    Returns:
        A pre-trained denoiser.
    """

    card = load_cards(__name__)[name]

    with open(download(card.url, hash_prefix=card.hash), "rb") as f:
        content = pickle.load(f)

    denoiser = content["ema"]
    denoiser = ElucidatedDenoiser(
        backbone=denoiser,
    )
    denoiser.eval()

    return denoiser
