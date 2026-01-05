r"""Just Image Transformer (JIT) plugin.

.. code-block:: python

    from azula.plugins import jit

References:
    | Back to Basics: Let Denoising Generative Models Denoise (Li et al., 2025)
    | https://arxiv.org/abs/2511.13720
"""

__all__ = [
    "JITDenoiser",
    "load_model",
]

import os
import torch
import torch.nn as nn

from torch import Tensor
from typing import Optional

from azula.denoise import Denoiser, DiracPosterior
from azula.hub import download
from azula.nn.utils import get_module_dtype, skip_init
from azula.noise import RectifiedSchedule, Schedule

from ._src.model import JiT_models
from ..utils import load_cards


class JITDenoiser(Denoiser):
    r"""Creates a JIT denoiser.

    Arguments:
        backbone: A time conditional network.
        schedule: A noise schedule. If :py:`None`, use
            :class:`azula.noise.RectifiedSchedule` instead.
        num_classes: The number of classes.
    """

    def __init__(
        self,
        backbone: nn.Module,
        schedule: Optional[Schedule] = None,
        num_classes: int = 1000,
    ):
        super().__init__()

        self.backbone = backbone

        if schedule is None:
            self.schedule = RectifiedSchedule()
        else:
            self.schedule = schedule

        self.num_classes = num_classes

    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        label: Optional[Tensor] = None,
        **kwargs,
    ) -> DiracPosterior:
        r"""
        Arguments:
            x_t: A noisy tensor :math:`x_t`, with shape :math:`(B, 3, H, W)`.
            t: The time :math:`t`, with shape :math:`()` or :math:`(B)`.
            label: The class label :math:`c` as an integer, with shape :math:`(B)`.
            kwargs: Optional keyword arguments.

        Returns:
            The Dirac delta :math:`\delta(X - \mu_\phi(x_t \mid c))`.
        """

        alpha_t, sigma_t = self.schedule(t)

        while alpha_t.ndim < x_t.ndim:
            alpha_t, sigma_t = alpha_t[..., None], sigma_t[..., None]

        c_in = 1 / (alpha_t + sigma_t)
        c_time = (alpha_t / (alpha_t + sigma_t)).flatten()

        B, _, _, _ = x_t.shape

        dtype = get_module_dtype(self.backbone)

        if label is None:
            label = torch.as_tensor(self.num_classes, device=x_t.device)

        output = self.backbone(
            (c_in * x_t).to(dtype),
            c_time.to(dtype),
            y=label.expand(B),
            **kwargs,
        ).to(x_t)

        mean = output

        return DiracPosterior(mean=mean)


def load_model(
    name: str,
    ema: bool = True,
    **kwargs,
) -> Denoiser:
    r"""Loads a pre-trained JIT denoiser.

    Arguments:
        name: The pre-trained model name.
        ema: Whether to load EMA weights or not.
        kwargs: Keyword arguments passed to :func:`torch.load`.

    Returns:
        A pre-trained denoiser.
    """

    kwargs.setdefault("map_location", "cpu")
    kwargs.setdefault("weights_only", True)

    card = load_cards(__name__)[name]

    state = torch.load(
        os.path.join(
            download(card.url, hash_prefix=card.hash, extract=True),
            "checkpoint-last.pth",
        ),
        **kwargs,
    )

    if ema:
        state = state["model_ema1"]
    else:
        state = state["model"]

    state = {k.removeprefix("net."): v for k, v in state.items()}

    with skip_init():
        denoiser = make_model(**card.config)

    denoiser.backbone.load_state_dict(state)

    return denoiser.eval()


def make_model(model: str = "JiT-B/16", **kwargs) -> Denoiser:
    r"""Initializes a JIT denoiser."""

    backbone = JiT_models[model](**kwargs)

    return JITDenoiser(
        backbone,
        num_classes=backbone.num_classes,
    )
