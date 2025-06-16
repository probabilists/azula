r"""Flux plugin.

This plugin depends on :mod:`diffusers` and :mod:`transformers`. To use it,
install the dependencies in your environment

.. code-block:: console

    pip install diffusers transformers accelerate protobuf sentencepiece

before importing the plugin.

.. code-block:: python

    from azula.plugins import flux

References:
    | FLUX (Black Forest Labs, 2024)
    | https://github.com/black-forest-labs/flux
"""

__all__ = [
    "AutoEncoder",
    "TextEncoder",
    "FluxDenoiser",
    "load_model",
]

import torch
import torch.nn as nn

from einops import rearrange
from functools import cache
from torch import Tensor
from typing import Dict, Optional, Sequence, Tuple, Union

from azula.denoise import Gaussian, GaussianDenoiser
from azula.nn.utils import skip_init
from azula.noise import DecaySchedule, Schedule

from ..utils import as_dtype, load_cards


class AutoEncoder(nn.Module):
    r"""Creates an auto-encoder wrapper."""

    def __init__(
        self,
        vae: nn.Module,  # AutoencoderKL
        shift: float = 0.0,
        scale: float = 1.0,
    ):
        super().__init__()

        self.vae = vae
        self.shift = shift
        self.scale = scale

    def encode(self, x: Tensor) -> Tensor:
        r"""Encodes images to latents.

        Arguments:
            x: A batch of images :math:`x`, with shape :math:`(B, 3, H, W)`.
                Pixel values are expected to range between -1 and 1.

        Returns:
            A batch of latents :math:`z \sim q(Z \mid x)`, with shape :math:`(B, H / 16, W / 16, 64)`.
        """

        dtype = {"dtype": self.vae.dtype, "device": self.vae.device}

        q_z_x = self.vae.encode(x.to(**dtype)).latent_dist
        z = torch.normal(q_z_x.mean, q_z_x.std)
        z = (z - self.shift) * self.scale
        z = rearrange(z, "... C (H h) (W w) -> ... H W (C h w)", h=2, w=2)

        return z.to(x)

    def decode(self, z: Tensor) -> Tensor:
        r"""Decodes latents to images.

        Arguments:
            z: A batch of latents :math:`z`, with shape :math:`(B, H / 16, W / 16, 64)`.

        Returns:
            A batch of images :math:`x = D(z)`, with shape :math:`(B, 3, H, W)`.
        """

        dtype = {"dtype": self.vae.dtype, "device": self.vae.device}

        z = rearrange(z, "... H W (C h w) -> ... C (H h) (W w)", h=2, w=2)
        z = z / self.scale + self.shift
        x = self.vae.decode(z.to(**dtype)).sample

        return x.to(z)


class TextEncoder(nn.Module):
    r"""Creates a text encoder."""

    def __init__(
        self,
        clip: nn.Module,
        clip_tokenizer: nn.Module,
        t5: nn.Module,
        t5_tokenizer: nn.Module,
    ):
        super().__init__()

        self.clip = clip
        self.clip_tokenizer = clip_tokenizer
        self.t5 = t5
        self.t5_tokenizer = t5_tokenizer

    def forward(self, prompt: Union[str, Sequence[str]]) -> Dict[str, Tensor]:
        r"""
        Arguments:
            prompt: A text prompt or list of text prompts.

        Returns:
            The CLIP and T5 encoded prompt(s).
        """

        if isinstance(prompt, str):
            prompt = [prompt]

        # CLIP
        clip_output = self.clip(
            input_ids=self.clip_tokenizer(
                prompt,
                truncation=True,
                max_length=self.clip_tokenizer.model_max_length,
                padding="max_length",
                return_tensors="pt",
            ).input_ids.to(device=self.clip.device),
            output_hidden_states=False,
        ).pooler_output

        # T5
        t5_output = self.t5(
            input_ids=self.t5_tokenizer(
                prompt,
                truncation=True,
                max_length=self.t5_tokenizer.model_max_length,
                padding="max_length",
                return_tensors="pt",
            ).input_ids.to(device=self.t5.device),
            output_hidden_states=False,
        ).last_hidden_state

        return {
            "prompt_clip": clip_output,
            "prompt_t5": t5_output,
        }


class FluxDenoiser(GaussianDenoiser):
    r"""Creates a Flux denoiser.

    Arguments:
        backbone: A time conditional network.
        schedule: A noise schedule. If :py:`None`, use
            :class:`azula.noise.DecaySchedule` instead.
    """

    def __init__(
        self,
        backbone: nn.Module,
        schedule: Optional[Schedule] = None,
    ):
        super().__init__()

        self.backbone = backbone

        if schedule is None:
            self.schedule = DecaySchedule()
        else:
            self.schedule = schedule

    @staticmethod
    @cache
    def coordinates(
        H: int,
        W: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        return torch.stack(
            torch.meshgrid(
                torch.zeros(1, dtype=dtype, device=device),
                torch.arange(H, dtype=dtype, device=device),
                torch.arange(W, dtype=dtype, device=device),
                indexing="ij",
            ),
            dim=-1,
        ).reshape(-1, 3)

    def forward(
        self,
        z_t: Tensor,
        t: Tensor,
        prompt_clip: Tensor,
        prompt_t5: Tensor,
        guidance: Union[float, Tensor] = 4.0,
        **kwargs,
    ) -> Gaussian:
        r"""
        Arguments:
            z_t: A noisy tensor :math:`z_t`, with shape :math:`(B, H, W, 64)`.
            t: The time :math:`t`, with shape :math:`()` or :math:`(B)`.
            prompt_clip: The CLIP-encoded text prompt :math:`y`, with shape :math:`(B, F)`.
            prompt_t5: The T5-encoded text prompt :math:`y`, with shape :math:`(B, L, D)`.
            guidance: The guidance strength :math:`\omega \in \mathbb{R}`.
            kwargs: Optional keyword arguments.

        Returns:
            The Gaussian :math:`\mathcal{N}(Z \mid \mu_\phi(z_t \mid y), \Sigma_\phi(z_t \mid y)`.
        """

        alpha_t, sigma_t = self.schedule(t)

        while alpha_t.ndim < z_t.ndim:
            alpha_t, sigma_t = alpha_t[..., None], sigma_t[..., None]

        c_in = 1 / (alpha_t + sigma_t)
        c_out = -sigma_t / (alpha_t + sigma_t)
        c_skip = 1 / (alpha_t + sigma_t)
        c_time = (sigma_t / (alpha_t + sigma_t)).flatten()
        c_var = sigma_t**2 / (alpha_t**2 + sigma_t**2)

        B, H, W, C = z_t.shape
        _, L, D = prompt_t5.shape

        dtype = {"dtype": self.backbone.dtype, "device": self.backbone.device}

        img_ids = self.coordinates(H, W, **dtype)
        txt_ids = torch.zeros((L, 3), **dtype)

        if guidance is not None:
            guidance = torch.as_tensor(guidance, **dtype).expand(B)

        output = self.backbone(
            timestep=c_time.to(**dtype).expand(B),
            hidden_states=(c_in * z_t).to(**dtype).reshape(B, H * W, C),
            encoder_hidden_states=prompt_t5.to(**dtype).expand(B, L, D),
            pooled_projections=prompt_clip.to(**dtype),
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
            **kwargs,
        ).sample.reshape_as(z_t)

        mean = c_skip * z_t + c_out * output.to(z_t)
        var = c_var

        return Gaussian(mean=mean, var=var)


def load_model(
    name: str = "flux_1_dev",
    **kwargs,
) -> Tuple[GaussianDenoiser, AutoEncoder, TextEncoder]:
    r"""Loads a pre-trained Flux latent denoiser.

    Arguments:
        name: The pre-trained model name.
        kwargs: Keyword arguments passed to :func:`diffusers.FluxPipeline.from_pretrained`.

    Returns:
        A pre-trained latent denoiser and the corresponding auto-encoder and text encoder.
    """

    from diffusers import FluxPipeline
    from unittest.mock import patch

    card = load_cards(__name__)[name]

    with skip_init(), patch("transformers.models.t5.tokenization_t5_fast.logger"):
        pipe = FluxPipeline.from_pretrained(
            card.repo,
            torch_dtype=as_dtype(card.dtype),
            variant=card.variant,
            **kwargs,
        )

    denoiser = FluxDenoiser(backbone=pipe.transformer)

    autoencoder = AutoEncoder(
        vae=pipe.vae,
        shift=pipe.vae.config.shift_factor,
        scale=pipe.vae.config.scaling_factor,
    )

    textencoder = TextEncoder(
        clip=pipe.text_encoder,
        clip_tokenizer=pipe.tokenizer,
        t5=pipe.text_encoder_2,
        t5_tokenizer=pipe.tokenizer_2,
    )

    return denoiser, autoencoder, textencoder
