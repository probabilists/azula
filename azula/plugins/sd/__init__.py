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

from azula.denoise import Denoiser, DiracPosterior
from azula.nn.utils import get_module_dtype, skip_init
from azula.noise import Schedule, VPSchedule

from ..utils import as_dtype, load_cards, patch_diffusers


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

        dtype = get_module_dtype(self.vae)

        q_z_x = self.vae.encode(x.to(dtype)).latent_dist
        z = q_z_x.mean + q_z_x.std * torch.randn_like(q_z_x.mean)
        z = z * self.scale

        return z.to(x)

    def decode(self, z: Tensor) -> Tensor:
        r"""Decodes latents to images.

        Arguments:
            z: A batch of latents :math:`z`, with shape :math:`(B, 4, H / 8, W / 8)`.

        Returns:
            A batch of images :math:`x = D(z)`, with shape :math:`(B, 3, H, W)`.
        """

        dtype = get_module_dtype(self.vae)

        z = z / self.scale
        x = self.vae.decode(z.to(dtype)).sample

        return x.to(z)


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
            mask = tokens.attention_mask
        else:
            mask = None

        embeds = self.clip(
            input_ids=tokens.input_ids,
            attention_mask=mask,
            output_hidden_states=False,
        ).last_hidden_state

        return {
            "prompt_embeds": embeds,
        }


class StableDenoiser(Denoiser):
    r"""Creates a stable denoiser.

    Arguments:
        backbone: A time conditional network.
        sigmas: The discrete noise schedule used during training.
        schedule: A noise schedule. If :py:`None`, use
            :class:`azula.noise.VPSchedule` instead.
        prediction: The backbone prediction type.
    """

    def __init__(
        self,
        backbone: nn.Module,
        sigmas: Tensor,
        schedule: Optional[Schedule] = None,
        prediction: str = "epsilon",
    ):
        super().__init__()

        self.backbone = backbone
        self.prediction = prediction

        if schedule is None:
            self.schedule = VPSchedule(
                alpha_min=(1 - sigmas[-1].item() ** 2) ** 0.5,
                sigma_min=sigmas[0].item(),
            )
        else:
            self.schedule = schedule

        self.register_buffer("sigmas", sigmas.to(torch.get_default_dtype()))

    def forward(
        self,
        z_t: Tensor,
        t: Tensor,
        prompt_embeds: Tensor,
        **kwargs,
    ) -> DiracPosterior:
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
        c_time = torch.searchsorted(self.sigmas, c_time.flatten())

        B, _, _, _ = z_t.shape
        _, L, D = prompt_embeds.shape

        dtype = get_module_dtype(self.backbone)

        output = self.backbone(
            timestep=c_time.expand(B),
            sample=(c_in * z_t).to(dtype),
            encoder_hidden_states=prompt_embeds.to(dtype).expand(B, L, D),
            **kwargs,
        ).sample.to(z_t)

        mean = c_skip * z_t + c_out * output

        return DiracPosterior(mean=mean)


def load_model(
    name: str,
    **kwargs,
) -> Tuple[Denoiser, AutoEncoder, TextEncoder]:
    r"""Loads a pre-trained stable latent denoiser.

    Arguments:
        name: The pre-trained model name.
        kwargs: Keyword arguments passed to :func:`diffusers.StableDiffusionPipeline.from_pretrained`.

    Returns:
        A pre-trained latent denoiser and the corresponding auto-encoder and text encoder.
    """

    from diffusers import StableDiffusionPipeline

    card = load_cards(__name__)[name]

    with skip_init(), patch_diffusers():
        pipe = StableDiffusionPipeline.from_pretrained(
            card.repo,
            torch_dtype=as_dtype(card.dtype),
            variant=card.variant,
            **kwargs,
        )

    alphas = pipe.scheduler.alphas_cumprod.to(dtype=torch.float64).sqrt()
    sigmas = torch.sqrt(1 - alphas**2)

    denoiser = StableDenoiser(
        backbone=pipe.unet,
        sigmas=sigmas,
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

    return denoiser.eval(), autoencoder.eval(), textencoder.eval()
