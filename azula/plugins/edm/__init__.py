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
    "model_cards",
    "load_model",
]

import os
import pickle
import torch
import torch.nn as nn
import yaml

from torch import Tensor
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

from azula.denoise import Gaussian, GaussianDenoiser
from azula.hub import download
from azula.noise import Schedule


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

    def forward(self, x_t: Tensor, t: Tensor, **kwargs) -> Gaussian:
        alpha_t, sigma_t = self.schedule(t)
        sigma_t, x_t = sigma_t / alpha_t, x_t / alpha_t

        kwargs.setdefault("class_labels", kwargs.pop("label", None))

        mean = self.backbone(x_t, sigma_t, **kwargs)

        while sigma_t.ndim < x_t.ndim:
            sigma_t = sigma_t[..., None]

        var = sigma_t**2 / (1 + sigma_t**2)

        return Gaussian(mean=mean, var=var)


def model_cards() -> Dict[str, SimpleNamespace]:
    r"""Returns a key-card mapping of available pre-trained models."""

    file = os.path.join(os.path.dirname(__file__), "cards.yml")

    with open(file, mode="r") as f:
        cards = yaml.safe_load(f)

    return {key: SimpleNamespace(**card) for key, card in cards.items()}


def load_model(key: str) -> GaussianDenoiser:
    r"""Loads a pre-trained EDM denoiser.

    Arguments:
        key: The pre-trained model key.

    Returns:
        A pre-trained denoiser.
    """

    card = model_cards()[key]

    with open(download(card.url, hash_prefix=card.hash), "rb") as f:
        content = pickle.load(f)

    denoiser = content["ema"]
    denoiser = ElucidatedDenoiser(
        backbone=denoiser,
    )
    denoiser.eval()

    return denoiser
