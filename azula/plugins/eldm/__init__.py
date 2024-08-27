r"""Elucidated latent diffusion model (ELDM or EDM2) plugin.

This plugin depends on the `torch_utils` and `training` modules in the `NVlabs/edm2
<https://github.com/NVlabs/edm2>`_ repository. To use it, clone the repository to you
machine

.. code-block:: console

    git clone https://github.com/NVlabs/edm2

and add it to your Python path before importing the plugin.

.. code-block:: python

    import sys; sys.path.append("path/to/edm2")
    ...
    from azula.plugins import eldm

You may also need to install additional dependencies, including :mod:`diffusers` and
:mod:`accelerate`.

.. code-block:: console

    pip install diffusers accelerate

References:
    | Analyzing and Improving the Training Dynamics of Diffusion Models (Karras et al., 2024)
    | https://arxiv.org/abs/2312.02696
"""

__all__ = [
    "AutoEncoder",
    "ElucidatedLatentDenoiser",
    "list_models",
    "load_model",
]

import pickle
import torch
import torch.nn as nn

from azula.denoise import Gaussian, GaussianDenoiser
from azula.hub import download
from azula.nn.utils import FlattenWrapper
from azula.noise import VESchedule
from diffusers.models import AutoencoderKL
from torch import Tensor
from typing import List, Optional, Tuple

# isort: split
from . import database
from ..edm import ElucidatedSchedule


class AutoEncoder(nn.Module):
    r"""Creates a standardized auto-encoder.

    Arguments:
        vae: A (variational) auto-encoder.
        shift: The shift to apply to latents, with shape :math:`(C, 1, 1)`.
        scale: The scale to apply to latents, with shape :math:`(C, 1, 1)`.
    """

    def __init__(
        self,
        vae: AutoencoderKL,
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

    .. math::
        \mu_\phi(x_t \mid c) & = (1 - \omega) \, b_\phi(x_t, \sigma_t)
            + \omega \, b_\phi(x_t, \sigma_t \mid c)  \\
        \sigma^2_\phi(x_t \mid c) & = \frac{\sigma_t^2}{1 + \sigma_t^2}

    where :math:`\omega \in \mathbb{R}_+` is the classifier-free guidance strength.

    Arguments:
        backbone: A noise conditional network :math:`b_\phi(x_t, \sigma_t \mid c)`.
        schedule: A variance exploding (VE) schedule. If :py:`None`, use
            :class:`azula.plugins.edm.ElucidatedSchedule` instead.
    """

    def __init__(
        self,
        backbone: nn.Module,
        schedule: Optional[VESchedule] = None,
    ):
        super().__init__()

        self.backbone = backbone

        if schedule is None:
            self.schedule = ElucidatedSchedule()
        else:
            self.schedule = schedule

    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        label: Optional[Tensor] = None,
        omega: Optional[Tensor] = None,
        **kwargs,
    ) -> Gaussian:
        r"""
        Arguments:
            x_t: A noisy vector :math:`x_t`, with shape :math:`(*, D)`.
            t: The time :math:`t`, with shape :math:`(*)`.
            label: The class label :math:`c` as a one-hot vector.
            omega: The classifier-free guidance strength :math:`\omega \in \mathbb{R}`.
                If :py:`None`, classifier-free guidance is not applied.
            kwargs: Optional keyword arguments.

        Returns:
            The Gaussian :math:`\mathcal{N}(X \mid \mu_\phi(x_t \mid c), \Sigma_\phi(x_t \mid c))`.
        """

        _, sigma_t = self.schedule(t)  # alpha_t = 1

        if label is None:
            mean = self.backbone(x_t, sigma_t.squeeze(-1), **kwargs)
        elif omega is None:
            mean = self.backbone(x_t, sigma_t.squeeze(-1), class_labels=label, **kwargs)
        else:
            mean = self.backbone(x_t, sigma_t.squeeze(-1), **kwargs)
            mean_cond = self.backbone(x_t, sigma_t.squeeze(-1), class_labels=label, **kwargs)
            mean = mean + omega * (mean_cond - mean)

        var = sigma_t**2 / (1 + sigma_t**2)

        return Gaussian(mean=mean, var=var)


def list_models() -> List[str]:
    r"""Returns the list of available pre-trained models."""

    return database.keys()


def load_model(key: str) -> Tuple[GaussianDenoiser, AutoEncoder]:
    r"""Loads a pre-trained ELDM (or EDM2) latent denoiser.

    Arguments:
        key: The pre-trained model key.

    Returns:
        A pre-trained latent denoiser and the corresponding auto-encoder.
    """

    url = database.get(key)

    with open(download(url), "rb") as f:
        content = pickle.load(f)

    denoiser = content["ema"]
    denoiser = ElucidatedLatentDenoiser(
        backbone=FlattenWrapper(
            wrappee=denoiser,
            shape=(4, 64, 64),
        ),
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
