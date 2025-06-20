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

from azula.denoise import Gaussian, GaussianDenoiser
from azula.hub import download
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

        q_z_x = self.vae.encode(x).latent_dist
        z = torch.normal(q_z_x.mean, q_z_x.std)
        z = z * self.scale + self.shift

        return z

    def decode(self, z: Tensor) -> Tensor:
        r"""Decodes latents to images.

        Arguments:
            z: A batch of latents :math:`z`, with shape :math:`(B, 4, 64, 64)`.

        Returns:
            A batch of images :math:`x = d(z)`, with shape :math:`(B, 3, 512, 512)`.
        """

        z = (z - self.shift) / self.scale
        x = self.vae.decode(z).sample

        return x


class ElucidatedLatentDenoiser(GaussianDenoiser):
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
    ) -> Gaussian:
        r"""
        Arguments:
            z_t: A noisy tensor :math:`z_t`, with shape :math:`(B, 4, 64, 64)`.
            t: The time :math:`t`, with shape :math:`()` or :math:`(B)`.
            label: The class label :math:`c` as a one-hot vector, with shape :math:`(*, 1000)`.
            kwargs: Optional keyword arguments.

        Returns:
            The Gaussian :math:`\mathcal{N}(Z \mid \mu_\phi(z_t \mid c), \Sigma_\phi(z_t \mid c))`.
        """

        alpha_t, sigma_t = self.schedule(t)

        while alpha_t.ndim < z_t.ndim:
            alpha_t, sigma_t = alpha_t[..., None], sigma_t[..., None]

        c_in = 1 / alpha_t
        c_time = (sigma_t / alpha_t).reshape_as(t)
        c_var = sigma_t**2 / (alpha_t**2 + sigma_t**2)

        mean = self.backbone(c_in * z_t, c_time, class_labels=label, **kwargs)
        var = c_var

        return Gaussian(mean=mean, var=var)


def load_model(name: str) -> Tuple[GaussianDenoiser, AutoEncoder]:
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
