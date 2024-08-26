r"""Elucidated diffusion model (EDM) plugin.

This plugin depends on the `dnnlib`, `torch_utils` and `training` modules in the
`NVlabs/edm <https://github.com/NVlabs/edm>`_ repository. To use it, clone the
repository to you machine

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
    "ElucidatedDenoiser",
    "list_models",
    "load_model",
]

import pickle
import re
import torch.nn as nn

from azula.denoise import Gaussian, GaussianDenoiser
from azula.hub import download
from azula.nn.utils import FlattenWrapper
from azula.noise import VESchedule
from torch import Tensor
from typing import List, Optional

# isort: split
from . import database


class ElucidatedDenoiser(GaussianDenoiser):
    r"""Creates an elucidated denoiser.

    Arguments:
        backbone: A noise conditional network.
        schedule: A variance exploding (VE) schedule.
    """

    def __init__(self, backbone: nn.Module, schedule: Optional[VESchedule] = None):
        super().__init__()

        self.backbone = backbone

        if schedule is None:
            self.schedule = VESchedule()
        else:
            self.schedule = schedule

    def forward(self, x_t: Tensor, t: Tensor, **kwargs) -> Gaussian:
        _, sigma_t = self.schedule(t)  # alpha_t = 1

        mean = self.backbone(x_t, sigma_t.squeeze(-1), **kwargs)
        var = sigma_t**2 / (1 + sigma_t**2)

        return Gaussian(mean=mean, var=var)


def list_models() -> List[str]:
    r"""Returns the list of available pre-trained models."""

    return database.keys()


def load_model(key: str) -> GaussianDenoiser:
    r"""Loads a pre-trained EDM model.

    Arguments:
        key: The pre-trained model key.

    Returns:
        A pre-trained denoiser.
    """

    url = database.get(key)
    filename = download(url)

    with open(filename, "rb") as f:
        model = pickle.load(f)["ema"]
        model.eval()

    image_size = re.search(r"(\d+)x(\d+)", key).groups()
    image_size = map(int, image_size)

    return ElucidatedDenoiser(
        backbone=FlattenWrapper(
            wrappee=model,
            shape=(3, *image_size),
        ),
    )
