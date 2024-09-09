r"""Velocity diffusion model (VDM) plugin.

This plugin depends on the `diffusion` module in the `crowsonkb/v-diffusion-pytorch
<https://github.com/crowsonkb/v-diffusion-pytorch>`_ repository. To use it, clone the
repository to you machine

.. code-block:: console

    git clone https://github.com/crowsonkb/v-diffusion-pytorch

and add it to your Python path before importing the plugin.

.. code-block:: python

    import sys; sys.path.append("path/to/v-diffusion-pytorch")
    ...
    from azula.plugins import vdm
"""

__all__ = [
    "AngularSchedule",
    "VelocityDenoiser",
    "list_models",
    "load_model",
]

import math
import torch
import torch.nn as nn

from azula.denoise import Gaussian, GaussianDenoiser
from azula.hub import download
from azula.nn.utils import FlattenWrapper
from azula.noise import Schedule
from azula.plugins.utils import RaiseMock
from torch import Tensor
from typing import List, Tuple

try:
    from diffusion.models import get_model  # type: ignore
except ImportError as e:
    get_model = RaiseMock(name="diffusion.models.get_model", error=e)

# isort: split
from . import database


class AngularSchedule(Schedule):
    r"""Creates an angular schedule."""

    def __init__(self):
        super().__init__()

        self.register_buffer("pi", torch.as_tensor(math.pi))

    def alpha(self, t: Tensor) -> Tensor:
        return torch.cos(self.pi / 2 * t)

    def sigma(self, t: Tensor) -> Tensor:
        return torch.sin(self.pi / 2 * t)

    def forward(self, t: Tensor) -> Tuple[Tensor, Tensor]:
        return self.alpha(t).unsqueeze(-1), self.sigma(t).unsqueeze(-1)


class VelocityDenoiser(GaussianDenoiser):
    r"""Creates a velocity denoiser.

    Arguments:
        backbone: A time conditional network.
        schedule: A noise schedule.
    """

    def __init__(self, backbone: nn.Module, schedule: Schedule):
        super().__init__()

        self.backbone = backbone
        self.schedule = schedule

    def forward(self, x_t: Tensor, t: Tensor, **kwargs) -> Gaussian:
        alpha_t, sigma_t = self.schedule(t)

        mean = alpha_t * x_t - sigma_t * self.backbone(x_t, t, **kwargs)
        var = sigma_t**2 / (alpha_t**2 + sigma_t**2)

        return Gaussian(mean=mean, var=var)


def list_models() -> List[str]:
    r"""Returns the list of available pre-trained models."""

    return database.keys()


def load_model(key: str, **kwargs) -> GaussianDenoiser:
    r"""Loads a pre-trained VDM denoiser.

    Arguments:
        key: The pre-trained model key.
        kwargs: Keyword arguments passed to :func:`torch.load`.

    Returns:
        A pre-trained denoiser.
    """

    kwargs.setdefault("map_location", "cpu")
    kwargs.setdefault("weights_only", True)

    url, hash_prefix, config = database.get(key)
    state = torch.load(download(url, hash_prefix=hash_prefix), **kwargs)

    denoiser = make_model(**config)
    denoiser.backbone.wrappee.load_state_dict(state)
    denoiser.eval()

    return denoiser


def make_model(
    # Data
    image_channels: int = 3,
    image_size: int = 128,
    # Backbone
    key: str = "imagenet_128",
) -> GaussianDenoiser:
    r"""Initializes a VDM denoiser."""

    backbone = FlattenWrapper(
        wrappee=get_model(key)(),
        shape=(image_channels, image_size, image_size),
    )

    schedule = AngularSchedule()

    return VelocityDenoiser(backbone, schedule)
