r"""Ablated diffusion model (ADM) plugin.

.. code-block:: python

    from azula.plugins import adm

References:
    | Diffusion Models Beat GANs on Image Synthesis (Dhariwal et al., 2021)
    | https://arxiv.org/abs/2105.05233
"""

__all__ = [
    "AblatedDenoiser",
    "load_model",
]

import torch
import torch.nn as nn

from torch import Tensor
from typing import Optional, Sequence

from azula.denoise import Denoiser, GaussianPosterior
from azula.hub import download
from azula.nn.utils import get_module_dtype, skip_init
from azula.noise import Schedule, VPSchedule

from ._src import unet
from ..utils import load_cards


class AblatedDenoiser(Denoiser):
    r"""Creates an ablated denoiser.

    Arguments:
        backbone: A time conditional network.
        schedule: A noise schedule. If :py:`None`, use :class:`azula.noise.VPSchedule`
            instead.
        clip_mean: Whether the mean :math:`\mu_\phi(x_t)` is clipped to :math:`[-1, 1]`
            or not during evaluation.
        learn_var: Whether the variance :math:`\sigma^2_\phi(x_t)` is learned or not.
            For pre-trained models, the learned variance is indicative, but inexact.
    """

    def __init__(
        self,
        backbone: nn.Module,
        schedule: Optional[Schedule] = None,
        clip_mean: bool = False,
        learn_var: bool = False,
        discrete_schedule: str = "linear",
        discrete_steps: int = 1000,
    ):
        super().__init__()

        self.backbone = backbone

        if schedule is None:
            self.schedule = VPSchedule(alpha_min=1e-2, sigma_min=1e-2)
        else:
            self.schedule = schedule

        self.clip_mean = clip_mean
        self.learn_var = learn_var

        if discrete_schedule == "linear":
            beta = torch.linspace(
                0.1 / discrete_steps,
                20.0 / discrete_steps,
                discrete_steps,
                dtype=torch.float64,
            )
        elif discrete_schedule == "cosine":
            t = torch.linspace(0, 1, discrete_steps + 1, dtype=torch.float64)
            alpha_bar = torch.cos((t + 0.008) / 1.008 * torch.pi / 2) ** 2
            beta = 1 - alpha_bar[1:] / alpha_bar[:-1]
            beta = torch.clip(beta, max=0.999)
        else:
            raise ValueError(f"Unknown discrete schedule '{discrete_schedule}'.")

        alpha_bar = torch.cumprod(1 - beta, dim=0)
        sigmas = torch.sqrt(1 - alpha_bar)

        self.register_buffer("sigmas", sigmas.to(torch.get_default_dtype()))

    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        label: Optional[Tensor] = None,
        **kwargs,
    ) -> GaussianPosterior:
        r"""
        Arguments:
            x_t: A noisy tensor :math:`x_t`, with shape :math:`(B, 3, H, W)`.
            t: The time :math:`t`, with shape :math:`()` or :math:`(B)`.
            label: The class label :math:`c` as an integer, with shape :math:`(B)`.
            kwargs: Optional keyword arguments.

        Returns:
            The Gaussian :math:`\mathcal{N}(X \mid \mu_\phi(x_t \mid c), \sigma^2_\phi(x_t \mid c))`.
        """

        alpha_t, sigma_t = self.schedule(t)

        while alpha_t.ndim < x_t.ndim:
            alpha_t, sigma_t = alpha_t[..., None], sigma_t[..., None]

        c_in = torch.rsqrt(alpha_t**2 + sigma_t**2)
        c_out = -sigma_t / alpha_t
        c_skip = 1 / alpha_t
        c_time = sigma_t * torch.rsqrt(alpha_t**2 + sigma_t**2)
        c_time = torch.searchsorted(self.sigmas, c_time.flatten())
        c_var = sigma_t**2 / (alpha_t**2 + sigma_t**2)

        dtype = get_module_dtype(self.backbone)

        output = self.backbone(
            (c_in * x_t).to(dtype),
            c_time,
            y=label,
            **kwargs,
        ).to(x_t)

        if self.learn_var:
            output, log_var = torch.chunk(output, 2, dim=1)
            mean = c_skip * x_t + c_out * output
            var = c_var * torch.exp(log_var)
        else:
            mean = c_skip * x_t + c_out * output
            var = c_var

        if not self.training and self.clip_mean:
            mean = torch.clip(mean, min=-1.0, max=1.0)

        return GaussianPosterior(mean=mean, var=var)


def load_model(name: str, **kwargs) -> Denoiser:
    r"""Loads a pre-trained ADM denoiser.

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
    # Denoiser
    clip_mean: bool = True,
    learn_var: bool = True,
    # Discrete schedule
    discrete_schedule: str = "linear",
    discrete_steps: int = 1000,
    # Data
    image_channels: int = 3,
    image_size: int = 64,
    # Backbone
    attention_resolutions: Sequence[int] = (32, 16, 8),
    channel_mult: Sequence[int] = (1, 2, 3, 4),
    num_channels: int = 128,
    num_classes: int = None,
    **kwargs,
) -> Denoiser:
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

    return AblatedDenoiser(
        backbone,
        clip_mean=clip_mean,
        learn_var=learn_var,
        discrete_schedule=discrete_schedule,
        discrete_steps=discrete_steps,
    )
