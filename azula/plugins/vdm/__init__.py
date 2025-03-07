r"""Velocity diffusion model (VDM) plugin.

This plugin depends on the `diffusion` module in the `crowsonkb/v-diffusion-pytorch
<https://github.com/crowsonkb/v-diffusion-pytorch>`_ repository. To use it, clone the
repository to your machine

.. code-block:: console

    git clone https://github.com/crowsonkb/v-diffusion-pytorch

and add it to your Python path before importing the plugin.

.. code-block:: python

    import sys; sys.path.append("path/to/v-diffusion-pytorch")
    ...
    from azula.plugins import vdm
"""

__all__ = [
    "CrowsonSchedule",
    "VelocityDenoiser",
    "model_cards",
    "load_model",
]

import os
import torch
import torch.nn as nn
import yaml

from torch import Tensor
from types import SimpleNamespace
from typing import Dict, Tuple

from azula.debug import RaiseMock
from azula.denoise import Gaussian, GaussianDenoiser
from azula.hub import download
from azula.noise import Schedule

try:
    import diffusion as crowson  # type: ignore
except ImportError as e:
    crowson = RaiseMock(name="diffusion", error=e)


class CrowsonSchedule(Schedule):
    r"""Creates an angular noise schedule."""

    def __init__(self, spliced: bool = False):
        super().__init__()

        self.spliced = spliced

    def backbone_time(self, t: Tensor) -> Tensor:
        if self.spliced:
            return crowson.utils.get_spliced_ddpm_cosine_schedule(t)
        else:
            return crowson.utils.get_ddpm_schedule(t)

    def forward(self, t: Tensor) -> Tuple[Tensor, Tensor]:
        return crowson.utils.t_to_alpha_sigma(self.backbone_time(t))


class VelocityDenoiser(GaussianDenoiser):
    r"""Creates a velocity denoiser.

    Arguments:
        backbone: A time conditional network.
        schedule: A noise schedule.
    """

    def __init__(self, backbone: nn.Module, schedule: CrowsonSchedule):
        super().__init__()

        self.backbone = backbone
        self.schedule = schedule

    def forward(self, x_t: Tensor, t: Tensor, **kwargs) -> Gaussian:
        alpha_t, sigma_t = self.schedule(t)

        while alpha_t.ndim < x_t.ndim:
            alpha_t, sigma_t = alpha_t[..., None], sigma_t[..., None]

        mean = alpha_t * x_t - sigma_t * self.backbone(
            x_t, self.schedule.backbone_time(t), **kwargs
        )
        var = sigma_t**2 / (alpha_t**2 + sigma_t**2)

        return Gaussian(mean=mean, var=var)


def model_cards() -> Dict[str, SimpleNamespace]:
    r"""Returns a key-card mapping of available pre-trained models."""

    file = os.path.join(os.path.dirname(__file__), "cards.yml")

    with open(file, mode="r") as f:
        cards = yaml.safe_load(f)

    return {key: SimpleNamespace(**card) for key, card in cards.items()}


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

    card = model_cards()[key]
    state = torch.load(download(card.url, hash_prefix=card.hash), **kwargs)

    denoiser = make_model(**card.config)
    denoiser.backbone.load_state_dict(state)
    denoiser.eval()

    return denoiser


def make_model(
    model: str = "imagenet_128",
) -> GaussianDenoiser:
    r"""Initializes a VDM denoiser."""

    backbone = crowson.models.get_model(model)()
    schedule = CrowsonSchedule(spliced=backbone.min_t == 0)

    return VelocityDenoiser(backbone, schedule)
