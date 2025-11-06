r"""Elucidated latent diffusion model (ELDM or EDM2) plugin.

This plugin depends on the `torch_utils` and `training` modules in the `NVlabs/edm2
<https://github.com/NVlabs/edm2>`_ repository. To use it, clone the repository to your
machine

.. code-block:: console

    git clone https://github.com/NVlabs/edm2

and add it to your Python path before importing the plugin.

.. code-block:: python

    import sys; sys.path.append("path/to/edm2")
    ...
    from azula.plugins import eldm

You may also need to install additional dependencies in your environment, including
:mod:`diffusers` and :mod:`accelerate`.

.. code-block:: console

    pip install diffusers accelerate

References:
    | Analyzing and Improving the Training Dynamics of Diffusion Models (Karras et al., 2024)
    | https://arxiv.org/abs/2312.02696
"""

__all__ = [
    "AutoEncoder",
    "ElucidatedLatentDenoiser",
    "load_model",
]

import pickle
import torch
import torch.nn as nn

from torch import Tensor
from typing import Optional, Tuple

from azula.denoise import Denoiser, DiracPosterior
from azula.hub import download
from azula.nn.utils import get_module_dtype
from azula.noise import Schedule

from ..edm import ElucidatedSchedule
from ..utils import load_cards


class AutoEncoder(nn.Module):
    r"""Creates an auto-encoder wrapper."""

    def __init__(
        self,
        vae: nn.Module,  # AutoencoderKL
        shift: Tensor,
        scale: Tensor,
    ):
        super().__init__()

        self.vae = vae

        self.register_buffer("shift", torch.as_tensor(shift))
        self.register_buffer("scale", torch.as_tensor(scale))

    def encode(self, x: Tensor) -> Tensor:
        r"""Encodes images to latents.

        Arguments:
            x: A batch of images :math:`x`, with shape :math:`(B, 3, 512, 512)`.
                Pixel values are expected to range between 0 and 1.

        Returns:
            A batch of latents :math:`z \sim q(Z \mid x)`, with shape :math:`(B, 4, 64, 64)`.
        """

        dtype = get_module_dtype(self.vae)

        q_z_x = self.vae.encode(x.to(dtype)).latent_dist
        z = q_z_x.mean + q_z_x.std * torch.randn_like(q_z_x.mean)
        z = z * self.scale + self.shift

        return z.to(x)

    def decode(self, z: Tensor) -> Tensor:
        r"""Decodes latents to images.

        Arguments:
            z: A batch of latents :math:`z`, with shape :math:`(B, 4, 64, 64)`.

        Returns:
            A batch of images :math:`x = D(z)`, with shape :math:`(B, 3, 512, 512)`.
        """

        dtype = get_module_dtype(self.vae)

        z = (z - self.shift) / self.scale
        x = self.vae.decode(z.to(dtype)).sample

        return x.to(z)


class ElucidatedLatentDenoiser(Denoiser):
    r"""Creates an elucidated latent denoiser.

    Arguments:
        backbone: A noise conditional network.
        schedule: A noise schedule. If :py:`None`, use
            :class:`azula.plugins.edm.ElucidatedSchedule` instead.
    """

    def __init__(
        self,
        backbone: nn.Module,
        schedule: Optional[Schedule] = None,
    ):
        super().__init__()

        self.backbone = backbone

        if schedule is None:
            self.schedule = ElucidatedSchedule()
        else:
            self.schedule = schedule

    def forward(
        self,
        z_t: Tensor,
        t: Tensor,
        label: Optional[Tensor] = None,
        **kwargs,
    ) -> DiracPosterior:
        r"""
        Arguments:
            z_t: A noisy tensor :math:`z_t`, with shape :math:`(B, 4, 64, 64)`.
            t: The time :math:`t`, with shape :math:`()` or :math:`(B)`.
            label: The class label :math:`c` as a one-hot vector, with shape :math:`(*, 1000)`.
            kwargs: Optional keyword arguments.

        Returns:
            The Dirac delta :math:`\delta(Z - \mu_\phi(z_t \mid c))`.
        """

        alpha_t, sigma_t = self.schedule(t)

        while alpha_t.ndim < z_t.ndim:
            alpha_t, sigma_t = alpha_t[..., None], sigma_t[..., None]

        c_in = 1 / alpha_t
        c_time = (sigma_t / alpha_t).reshape_as(t)

        dtype = get_module_dtype(self.backbone)

        mean = self.backbone(
            (c_in * z_t).to(dtype),
            c_time.to(dtype),
            class_labels=label.to(dtype),
            **kwargs,
        ).to(z_t)

        return DiracPosterior(mean=mean)


def load_model(name: str) -> Tuple[Denoiser, AutoEncoder]:
    r"""Loads a pre-trained ELDM (or EDM2) latent denoiser.

    Arguments:
        name: The pre-trained model name.

    Returns:
        A pre-trained latent denoiser and the corresponding auto-encoder.
    """

    from diffusers import AutoencoderKL

    card = load_cards(__name__)[name]

    with open(download(card.url, hash_prefix=card.hash), "rb") as f:
        content = pickle.load(f)

    denoiser = content["ema"]
    denoiser = ElucidatedLatentDenoiser(
        backbone=denoiser,
    )
    denoiser.eval()

    autoencoder = content["encoder"]
    autoencoder = AutoEncoder(
        vae=AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse"),
        shift=autoencoder.bias.reshape(-1, 1, 1),
        scale=autoencoder.scale.reshape(-1, 1, 1),
    )
    autoencoder.eval()

    return denoiser, autoencoder
