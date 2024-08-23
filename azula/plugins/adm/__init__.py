r"""Ablated diffusion model (ADM) plugin.

This plugin depends on the `guided_diffusion` module in the `openai/guided-diffusion
<https://github.com/openai/guided-diffusion>`_ repository. To use it, clone the
repository to you machine

.. code-block:: console

    git clone https://github.com/openai/guided-diffusion

and add it to your Python path.

.. code-block:: python

    import sys; sys.path.append("path/to/guided-diffusion")

References:
    | Diffusion Models Beat GANs on Image Synthesis (Dhariwal et al., 2021)
    | https://arxiv.org/abs/2105.05233
"""

__all__ = [
    "BetaSchedule",
    "ImprovedDenoiser",
    "list_models",
    "load_model",
    "make_model",
]

import numpy as np
import torch
import torch.nn as nn

from azula.debug import RaiseMock
from azula.denoise import Gaussian, GaussianDenoiser
from azula.nn.utils import FlattenWrapper
from azula.noise import Schedule
from torch import LongTensor, Tensor
from typing import List, Sequence, Set, Tuple

try:
    from gdown import cached_download
except ImportError as e:
    cached_download = RaiseMock(name="gdown.cached_download", error=e)

try:
    from guided_diffusion import unet  # type: ignore
except ImportError as e:
    unet = RaiseMock(name="guided_diffusion.unet", error=e)

# isort: split
from . import database


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
            beta = np.clip(beta, a_max=0.999)
        else:
            raise ValueError(f"Unknown schedule name '{name}'.")

        alpha_bar = np.cumprod(1 - beta)

        alpha = np.sqrt(alpha_bar)
        sigma = np.sqrt(1 - alpha_bar)

        log_beta = np.log(beta)
        log_beta_tilde = log_beta + 2 * np.diff(np.log(sigma), prepend=0)
        log_beta_tilde[0] = log_beta_tilde[1]

        self.register_buffer("beta", torch.as_tensor(beta))
        self.register_buffer("alpha", torch.as_tensor(alpha))
        self.register_buffer("sigma", torch.as_tensor(sigma))
        self.register_buffer("log_beta", torch.as_tensor(log_beta))
        self.register_buffer("log_beta_tilde", torch.as_tensor(log_beta_tilde))

        self.to(dtype=torch.float32)

    @property
    def steps(self) -> int:
        return len(self.beta)

    def discrete(self, t: Tensor) -> LongTensor:
        return torch.round((self.steps - 1) * t).long()

    def forward(self, t: Tensor) -> Tuple[Tensor, Tensor]:
        t = self.discrete(t)

        alpha_t = self.alpha[t]
        sigma_t = self.sigma[t]

        return alpha_t.unsqueeze(-1), sigma_t.unsqueeze(-1)

    def kernel(self, t: LongTensor) -> Tuple[Tensor, Tensor]:
        alpha_t = torch.where(t < 0, 1, self.alpha[t])
        sigma_t = torch.where(t < 0, 0, self.sigma[t])

        return alpha_t.unsqueeze(-1), sigma_t.unsqueeze(-1)


class ImprovedDenoiser(GaussianDenoiser):
    r"""Creates an improved DDPM denoiser.

    References:
        | Improved Denoising Diffusion Probabilistic Models (Nichol et al., 2021)
        | https://arxiv.org/abs/2102.09672

    Arguments:
        backbone: A discrete time conditional network.
        schedule: A beta schedule.
    """

    def __init__(self, backbone: nn.Module, schedule: BetaSchedule):
        super().__init__()

        self.backbone = backbone
        self.schedule = schedule

    def forward(self, x_t: Tensor, t: Tensor, **kwargs) -> Gaussian:
        t = self.schedule.discrete(t)

        alpha_t, sigma_t = self.schedule.kernel(t)
        alpha_s, sigma_s = self.schedule.kernel(t - 1)

        output = self.backbone(x_t, t, **kwargs)

        if output.shape == x_t.shape:
            eps = output
            mean = (x_t - sigma_t * eps) / alpha_t
            var = sigma_t**2 / (alpha_t**2 + sigma_t**2)
        else:
            eps, var = torch.chunk(output, 2, dim=-1)
            mean = (x_t - sigma_t * eps) / alpha_t

            log_beta = self.schedule.log_beta[t].unsqueeze(-1)
            log_beta_tilde = self.schedule.log_beta_tilde[t].unsqueeze(-1)

            frac = (var + 1) / 2
            var_s_t = torch.exp(frac * (log_beta - log_beta_tilde) + log_beta_tilde)

            tau = 1 - (alpha_t / alpha_s * sigma_s / sigma_t) ** 2
            shift = sigma_s**2 * tau
            scale = alpha_s**2 * tau**2

            var = (var_s_t - shift) / scale

        return Gaussian(mean=mean, var=var)


def list_models() -> List[str]:
    r"""Returns the list of available pre-trained models."""

    return database.keys()


def load_model(key: str, **kwargs) -> ImprovedDenoiser:
    r"""Loads a pre-trained ADM model.

    Arguments:
        key: The pre-trained model key.
        kwargs: Keyword arguments passed to :func:`torch.hub.load`.

    Returns:
        A pre-trained denoiser.
    """

    kwargs.setdefault("map_location", "cpu")
    kwargs.setdefault("weights_only", True)

    url, config = database.get(key)

    if "drive.google" in url:
        state = torch.load(
            f=cached_download(url=url),
            **kwargs,
        )
    else:
        state = torch.hub.load_state_dict_from_url(
            url=url,
            **kwargs,
        )

    denoiser = make_model(**config)
    denoiser.backbone.wrappee.load_state_dict(state)
    denoiser.eval()

    return denoiser


def make_model(
    # Denoiser
    learned_var: bool = True,
    # Schedule
    schedule_name: str = "linear",
    timesteps: int = 1000,
    # Backbone
    attention_resolutions: Set[int] = {32, 16, 8},  # noqa: B006
    channel_mult: Sequence[int] = (1, 2, 3, 4),
    dropout: float = 0.0,
    image_size: int = 64,
    num_channels: int = 128,
    num_classes: int = None,
    num_heads: int = 1,
    num_head_channels: int = 64,
    num_res_blocks: int = 3,
    **kwargs,
) -> ImprovedDenoiser:
    r"""Builds an ADM model.

    Arguments:
        learned_var: Whether the variance term is learned or not.
        schedule_name: The beta schedule name.
        timesteps: The number of schedule time steps.

    The remaining arguments are for the :class:`guided_diffusion.unet.UNetModel`
    backbone.

    Returns:
        A denoiser.
    """

    kwargs.setdefault("resblock_updown", True)
    kwargs.setdefault("use_fp16", False)
    kwargs.setdefault("use_new_attention_order", False)
    kwargs.setdefault("use_scale_shift_norm", True)

    attention_resolutions = {image_size // r for r in attention_resolutions}

    backbone = FlattenWrapper(
        wrappee=unet.UNetModel(
            image_size=image_size,
            in_channels=3,
            out_channels=6 if learned_var else 3,
            model_channels=num_channels,
            channel_mult=channel_mult,
            num_classes=num_classes,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            dropout=dropout,
            **kwargs,
        ),
        shape=(3, image_size, image_size),
    )

    schedule = BetaSchedule(name=schedule_name, steps=timesteps)

    return ImprovedDenoiser(backbone=backbone, schedule=schedule)


# fmt: off
def monkey_checkpoint(func, inputs, params, flag):
    return func(*inputs)

unet.checkpoint = monkey_checkpoint
# fmt: on
