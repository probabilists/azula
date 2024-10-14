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
    "list_models",
    "load_model",
]

import pickle
import re
import torch
import torch.nn as nn

from torch import Tensor
from typing import List, Optional

from azula.denoise import Gaussian, GaussianDenoiser
from azula.hub import download
from azula.nn.utils import FlattenWrapper
from azula.noise import VESchedule

from . import database


class ElucidatedSchedule(VESchedule):
    r"""Creates an elucidated noise schedule.

    .. math::
        \alpha_t & = 1 \\
        \sigma_t & = \left( (1 - t) \, {\sigma_\min}^\frac{1}{\gamma}
            + t \, {\sigma_\max}^\frac{1}{\gamma} \right)^\gamma

    Arguments:
        sigma_min: The initial noise scale :math:`\sigma_\min \in \mathbb{R}_+`.
        sigma_max: The final noise scale :math:`\sigma_\max \in \mathbb{R}_+`.
        gamma: A hyper-parameter :math:`\gamma \in \mathbb{R}_+`, denoted :math:`\rho` originally.
    """

    def __init__(self, sigma_min: float = 0.002, sigma_max: float = 80.0, gamma: float = 7.0):
        super().__init__(sigma_min, sigma_max)

        self.register_buffer("gamma", torch.as_tensor(gamma))

    def sigma(self, t: Tensor) -> Tensor:
        lower = torch.exp(self.log_sigma_min / self.gamma)
        upper = torch.exp(self.log_sigma_max / self.gamma)

        return torch.lerp(lower, upper, t) ** self.gamma


class ElucidatedDenoiser(GaussianDenoiser):
    r"""Creates an elucidated denoiser.

    Arguments:
        backbone: A noise conditional network.
        schedule: A variance exploding (VE) schedule. If :py:`None`, use
            :class:`ElucidatedSchedule` instead.
    """

    def __init__(self, backbone: nn.Module, schedule: Optional[VESchedule] = None):
        super().__init__()

        self.backbone = backbone

        if schedule is None:
            self.schedule = ElucidatedSchedule()
        else:
            self.schedule = schedule

    def forward(self, x_t: Tensor, t: Tensor, **kwargs) -> Gaussian:
        _, sigma_t = self.schedule(t)  # alpha_t = 1

        kwargs.setdefault("class_labels", kwargs.get("label", None))

        mean = self.backbone(x_t, sigma_t.squeeze(-1), **kwargs)
        var = sigma_t**2 / (1 + sigma_t**2)

        return Gaussian(mean=mean, var=var)


def list_models() -> List[str]:
    r"""Returns the list of available pre-trained models."""

    return database.keys()


def load_model(key: str) -> GaussianDenoiser:
    r"""Loads a pre-trained EDM denoiser.

    Arguments:
        key: The pre-trained model key.

    Returns:
        A pre-trained denoiser.
    """

    url = database.get(key)

    with open(download(url), "rb") as f:
        content = pickle.load(f)

    denoiser = content["ema"]
    denoiser.eval()

    image_size = re.search(r"(\d+)x(\d+)", key).groups()
    image_size = map(int, image_size)

    return ElucidatedDenoiser(
        backbone=FlattenWrapper(
            wrappee=denoiser,
            shape=(3, *image_size),
        ),
    )
