r"""Stable Diffusion (SD) plugin.

This plugin depends on :mod:`diffusers` and :mod:`transformers`. To use it,
install the dependencies in your environment

.. code-block:: console

    pip install diffusers transformers accelerate

before importing the plugin.

.. code-block:: python

    from azula.plugins import sd

References:
    | High-Resolution Image Synthesis with Latent Diffusion Models (Rombach et al., 2021)
    | https://arxiv.org/abs/2112.10752
"""

__all__ = [
    "AutoEncoder",
    "TextEncoder",
    "StableDenoiser",
    "load_model",
]

import torch
import torch.nn as nn

from torch import Tensor
from typing import Dict, Optional, Sequence, Tuple, Union

from azula.denoise import Gaussian, GaussianDenoiser
from azula.nn.utils import skip_init
from azula.noise import Schedule, VPSchedule

from ..utils import as_dtype, load_cards


class AutoEncoder(nn.Module):
    r"""Creates an auto-encoder wrapper."""

    def __init__(
        self,
        vae: nn.Module,  # AutoencoderKL
        scale: float = 1.0,
    ):
        super().__init__()

        self.vae = vae
        self.scale = scale

    def encode(self, x: Tensor) -> Tensor:
        r"""Encodes images to latents.

        Arguments:
            x: A batch of images :math:`x`, with shape :math:`(B, 3, H, W)`.
                Pixel values are expected to range between 0 and 1.

        Returns:
            A batch of latents :math:`z \sim q(Z \mid x)`, with shape :math:`(B, 4, H / 8, W / 8)`.
        """

        dtype = {"dtype": self.vae.dtype, "device": self.vae.device}

        q_z_x = self.vae.encode(x.to(**dtype)).latent_dist
        z = torch.normal(q_z_x.mean, q_z_x.std)
        z = z * self.scale

        return z

    def decode(self, z: Tensor) -> Tensor:
        r"""Decodes latents to images.

        Arguments:
            z: A batch of latents :math:`z`, with shape :math:`(B, 4, H / 8, W / 8)`.

        Returns:
            A batch of images :math:`x = D(z)`, with shape :math:`(B, 3, H, W)`.
        """

        dtype = {"dtype": self.vae.dtype, "device": self.vae.device}

        z = z / self.scale
        x = self.vae.decode(z.to(**dtype)).sample

        return x


class TextEncoder(nn.Module):
    r"""Creates a text encoder."""

    def __init__(
        self,
        clip: nn.Module,
        tokenizer: nn.Module,
    ):
        super().__init__()

        self.clip = clip
        self.tokenizer = tokenizer

    def forward(self, prompt: Union[str, Sequence[str]]) -> Dict[str, Tensor]:
        r"""
        Arguments:
            prompt: A text prompt or list of text prompts.

        Returns:
            The CLIP encoded prompt(s).
        """

        if isinstance(prompt, str):
            prompt = [prompt]

        tokens = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            return_tensors="pt",
        )

        if getattr(self.clip.config, "use_attention_mask", False):
            mask = tokens.attention_mask.to(device=self.clip.device)
        else:
            mask = None

        embeds = self.clip(
            input_ids=tokens.input_ids.to(device=self.clip.device),
            attention_mask=mask,
            output_hidden_states=False,
        ).last_hidden_state

        return {
            "prompt_embeds": embeds,
        }


class StableDenoiser(GaussianDenoiser):
    r"""Creates a stable denoiser.

    Arguments:
        backbone: A time conditional network.
        discrete: The discrete noise schedule used during training.
        schedule: A noise schedule. If :py:`None`, use
            :class:`azula.noise.VPSchedule` instead.
        prediction: The backbone prediction type.
    """

    def __init__(
        self,
        backbone: nn.Module,
        discrete: Tensor,
        schedule: Optional[Schedule] = None,
        prediction: str = "epsilon",
    ):
        super().__init__()

        self.backbone = backbone
        self.prediction = prediction

        if schedule is None:
            self.schedule = VPSchedule(
                alpha_min=(1 - discrete[-1].item() ** 2) ** 0.5,
                sigma_min=discrete[0].item(),
            )
        else:
            self.schedule = schedule

        self.register_buffer("discrete", discrete)

    def forward(
        self,
        z_t: Tensor,
        t: Tensor,
        prompt_embeds: Tensor,
        **kwargs,
    ) -> Gaussian:
        r"""
        Arguments:
            z_t: A noisy tensor :math:`z_t`, with shape :math:`(B, C, H, W)`.
            t: The time :math:`t`, with shape :math:`()` or :math:`(B)`.
            prompt_embeds: The CLIP-encoded text prompt :math:`y`, with shape :math:`(B, L, D)`.
            kwargs: Optional keyword arguments.

        Returns:
            The Gaussian :math:`\mathcal{N}(Z \mid \mu_\phi(z_t \mid y), \Sigma_\phi(z_t \mid y)`.
        """

        alpha_t, sigma_t = self.schedule(t)

        while alpha_t.ndim < z_t.ndim:
            alpha_t, sigma_t = alpha_t[..., None], sigma_t[..., None]

        if self.prediction == "epsilon":
            c_out = -sigma_t / alpha_t
            c_skip = 1 / alpha_t
        elif self.prediction == "velocity":
            c_out = -sigma_t * torch.rsqrt(alpha_t**2 + sigma_t**2)
            c_skip = alpha_t * torch.rsqrt(alpha_t**2 + sigma_t**2)
        else:
            raise ValueError(f"Unkown prediction type '{self.prediction}'.")

        c_in = torch.rsqrt(alpha_t**2 + sigma_t**2)
        c_time = sigma_t * torch.rsqrt(alpha_t**2 + sigma_t**2)
        c_time = torch.searchsorted(self.discrete, c_time.flatten())
        c_var = sigma_t**2 / (alpha_t**2 + sigma_t**2)

        B, _, _, _ = z_t.shape
        _, L, D = prompt_embeds.shape

        dtype = {"dtype": self.backbone.dtype, "device": self.backbone.device}

        output = self.backbone(
            timestep=c_time.to(**dtype).expand(B),
            sample=(c_in * z_t).to(**dtype),
            encoder_hidden_states=prompt_embeds.to(**dtype).expand(B, L, D),
            **kwargs,
        ).sample

        mean = c_skip * z_t + c_out * output.to(z_t)
        var = c_var

        return Gaussian(mean=mean, var=var)


def load_model(
    name: str,
    **kwargs,
) -> Tuple[GaussianDenoiser, AutoEncoder, TextEncoder]:
    r"""Loads a pre-trained stable latent denoiser.

    Arguments:
        name: The pre-trained model name.
        kwargs: Keyword arguments passed to :func:`diffusers.StableDiffusionPipeline.from_pretrained`.

    Returns:
        A pre-trained latent denoiser and the corresponding auto-encoder and text encoder.
    """

    from diffusers import StableDiffusionPipeline

    card = load_cards(__name__)[name]

    with skip_init():
        pipe = StableDiffusionPipeline.from_pretrained(
            card.repo,
            torch_dtype=as_dtype(card.dtype),
            variant=card.variant,
            **kwargs,
        )

    alphas = pipe.scheduler.alphas_cumprod.to(dtype=torch.float64).sqrt()
    sigmas = torch.sqrt(1 - alphas**2).to(dtype=torch.float32)

    denoiser = StableDenoiser(
        backbone=pipe.unet,
        discrete=sigmas,
        **card.config,
    )

    autoencoder = AutoEncoder(
        vae=pipe.vae,
        scale=pipe.vae.config.scaling_factor,
    )

    textencoder = TextEncoder(
        clip=pipe.text_encoder,
        tokenizer=pipe.tokenizer,
    )

    return denoiser, autoencoder, textencoder
