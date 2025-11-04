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
    "VelocityDenoiser",
    "load_model",
]

import torch
import torch.nn as nn

from torch import Tensor
from typing import Optional

from azula.debug import RaiseMock
from azula.denoise import Denoiser, DiracPosterior
from azula.hub import download
from azula.nn.utils import get_module_dtype, skip_init
from azula.noise import Schedule, VPSchedule

from ..utils import load_cards

try:
    import diffusion as crowson  # type: ignore
except ImportError as e:
    crowson = RaiseMock(name="diffusion", error=e)


class VelocityDenoiser(Denoiser):
    r"""Creates a velocity denoiser.

    Arguments:
        backbone: A time conditional network.
        schedule: A noise schedule. If :py:`None`, use :class:`azula.noise.VPSchedule`
            instead.
    """

    def __init__(
        self,
        backbone: nn.Module,
        schedule: Optional[Schedule] = None,
    ):
        super().__init__()

        self.backbone = backbone

        if schedule is None:
            self.schedule = VPSchedule(alpha_min=1e-2, sigma_min=1e-2)
        else:
            self.schedule = schedule

    def forward(self, x_t: Tensor, t: Tensor, **kwargs) -> DiracPosterior:
        alpha_t, sigma_t = self.schedule(t)

        while alpha_t.ndim < x_t.ndim:
            alpha_t, sigma_t = alpha_t[..., None], sigma_t[..., None]

        c_in = torch.rsqrt(alpha_t**2 + sigma_t**2)
        c_out = -sigma_t * torch.rsqrt(alpha_t**2 + sigma_t**2)
        c_skip = alpha_t * torch.rsqrt(alpha_t**2 + sigma_t**2)
        c_time = crowson.utils.alpha_sigma_to_t(alpha_t, sigma_t).flatten()

        dtype = get_module_dtype(self.backbone)

        output = self.backbone(
            (c_in * x_t).to(dtype),
            c_time.to(dtype),
            **kwargs,
        ).to(x_t)

        mean = c_skip * x_t + c_out * output

        return DiracPosterior(mean=mean)


def load_model(name: str, **kwargs) -> Denoiser:
    r"""Loads a pre-trained VDM denoiser.

    Arguments:
        name: The pre-trained model name.
        kwargs: Keyword arguments passed to :func:`torch.load`.

    Returns:
        A pre-trained denoiser.
    """

    kwargs.setdefault("map_location", "cpu")
    kwargs.setdefault("weights_only", True)

    card = load_cards(__name__)[name]
    state = torch.load(download(card.url, hash_prefix=card.hash), **kwargs)

    with skip_init():
        denoiser = make_model(**card.config)

    denoiser.backbone.load_state_dict(state)

    return denoiser.eval()


def make_model(
    model: str = "imagenet_128",
) -> Denoiser:
    r"""Initializes a VDM denoiser."""

    backbone = crowson.models.get_model(model)()

    return VelocityDenoiser(backbone)
