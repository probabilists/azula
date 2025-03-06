r"""Ablated diffusion model (ADM) plugin.

This plugin depends on the `guided_diffusion` module in the `openai/guided-diffusion
<https://github.com/openai/guided-diffusion>`_ repository. To use it, clone the
repository to your machine

.. code-block:: console

    git clone https://github.com/openai/guided-diffusion

and add it to your Python path before importing the plugin.

.. code-block:: python

    import sys; sys.path.append("path/to/guided-diffusion")
    ...
    from azula.plugins import adm

References:
    | Diffusion Models Beat GANs on Image Synthesis (Dhariwal et al., 2021)
    | https://arxiv.org/abs/2105.05233
"""

__all__ = [
    "BetaSchedule",
    "ImprovedDenoiser",
    "model_cards",
    "load_model",
]

import numpy as np
import os
import torch
import torch.nn as nn
import yaml

from torch import LongTensor, Tensor
from types import SimpleNamespace
from typing import Dict, Sequence, Tuple

from azula.debug import RaiseMock
from azula.denoise import Gaussian, GaussianDenoiser
from azula.hub import download
from azula.noise import Schedule

try:
    from guided_diffusion import unet  # type: ignore
except ImportError as e:
    unet = RaiseMock(name="guided_diffusion.unet", error=e)


class BetaSchedule(Schedule):
    r"""Creates a named beta schedule.

    Arguments:
        name: The schedule name.
        steps: The number of steps.
    """

    def __init__(self, name: str = "linear", steps: int = 1000):
        super().__init__()

        if name == "linear":
            beta = np.linspace(0.1 / steps, 20.0 / steps, steps)
        elif name == "cosine":
            t = np.linspace(0, 1, steps + 1)
            alpha_bar = np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2
            beta = 1 - alpha_bar[1:] / alpha_bar[:-1]
            beta = np.minimum(beta, 0.999)
        else:
            raise ValueError(f"Unknown schedule name '{name}'.")

        alpha_sq = np.cumprod(1 - beta)

        alpha = np.sqrt(alpha_sq)
        sigma = np.sqrt(1 - alpha_sq)

        self.steps = steps

        self.register_buffer("alpha", torch.as_tensor(alpha))
        self.register_buffer("sigma", torch.as_tensor(sigma))

        self.to(dtype=torch.float32)

    def discrete(self, t: Tensor) -> LongTensor:
        return torch.round((self.steps - 1) * t).long()

    def forward(self, t: Tensor) -> Tuple[Tensor, Tensor]:
        t = self.discrete(t)

        alpha_t = self.alpha[t]
        sigma_t = self.sigma[t]

        return alpha_t, sigma_t


class ImprovedDenoiser(GaussianDenoiser):
    r"""Creates an improved DDPM denoiser.

    References:
        | Improved Denoising Diffusion Probabilistic Models (Nichol et al., 2021)
        | https://arxiv.org/abs/2102.09672

    Arguments:
        backbone: A discrete time conditional network.
        schedule: A beta schedule.
        clip_mean: Whether the mean :math:`\mu_\phi(x_t)` is clipped to :math:`[-1, 1]`
            or not during evaluation.
        learn_var: Whether the variance :math:`\Sigma_\phi(x_t)` is learned or not.
            For pre-trained models, the learned variance is indicative, but inexact.
    """

    def __init__(
        self,
        backbone: nn.Module,
        schedule: BetaSchedule,
        clip_mean: bool = False,
        learn_var: bool = False,
    ):
        super().__init__()

        self.backbone = backbone
        self.schedule = schedule

        self.clip_mean = clip_mean
        self.learn_var = learn_var

    def forward(self, x_t: Tensor, t: Tensor, **kwargs) -> Gaussian:
        alpha_t, sigma_t = self.schedule(t)

        while alpha_t.ndim < x_t.ndim:
            alpha_t, sigma_t = alpha_t[..., None], sigma_t[..., None]

        kwargs.setdefault("y", kwargs.pop("label", None))

        output = self.backbone(x_t, self.schedule.discrete(torch.atleast_1d(t)), **kwargs)

        if self.learn_var:
            eps, v = torch.chunk(output, 2, dim=1)
            mean = (x_t - sigma_t * eps) / alpha_t
            var = sigma_t**2 / (alpha_t**2 + sigma_t**2) * torch.exp(v)
        else:
            eps = output
            mean = (x_t - sigma_t * eps) / alpha_t
            var = sigma_t**2 / (alpha_t**2 + sigma_t**2)

        if not self.training and self.clip_mean:
            mean = torch.clip(mean, min=-1.0, max=1.0)

        return Gaussian(mean=mean, var=var)


def model_cards() -> Dict[str, SimpleNamespace]:
    r"""Returns a key-card mapping of available pre-trained models."""

    file = os.path.join(os.path.dirname(__file__), "cards.yml")

    with open(file, mode="r") as f:
        cards = yaml.safe_load(f)

    return {key: SimpleNamespace(**card) for key, card in cards.items()}


def load_model(key: str, **kwargs) -> GaussianDenoiser:
    r"""Loads a pre-trained ADM denoiser.

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
    # Denoiser
    clip_mean: bool = True,
    learn_var: bool = True,
    # Schedule
    schedule_name: str = "linear",
    timesteps: int = 1000,
    # Data
    image_channels: int = 3,
    image_size: int = 64,
    # Backbone
    attention_resolutions: Sequence[int] = (32, 16, 8),
    channel_mult: Sequence[int] = (1, 2, 3, 4),
    num_channels: int = 128,
    num_classes: int = None,
    **kwargs,
) -> GaussianDenoiser:
    r"""Initializes an ADM denoiser."""

    attention_resolutions = {image_size // r for r in attention_resolutions}

    backbone = unet.UNetModel(
        image_size=image_size,
        in_channels=image_channels,
        out_channels=2 * image_channels if learn_var else image_channels,
        model_channels=num_channels,
        channel_mult=channel_mult,
        num_classes=num_classes,
        attention_resolutions=attention_resolutions,
        **kwargs,
    )

    if kwargs.get("use_fp16", False):
        backbone.convert_to_fp16()

    schedule = BetaSchedule(name=schedule_name, steps=timesteps)

    return ImprovedDenoiser(
        backbone,
        schedule,
        clip_mean=clip_mean,
        learn_var=learn_var,
    )


# fmt: off
def monkey_checkpoint(func, inputs, params, flag):
    return func(*inputs)

unet.checkpoint = monkey_checkpoint
# fmt: on
