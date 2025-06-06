r"""Sana plugin.

This plugin depends on :mod:`diffusers` and :mod:`transformers`. To use it,
install the dependencies in your environment

.. code-block:: console

    pip install diffusers transformers accelerate

before importing the plugin.

.. code-block:: python

    from azula.plugins import sana

References:
    | SANA: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers (Xie et al., 2024)
    | https://arxiv.org/abs/2410.10629
"""

__all__ = [
    "AutoEncoder",
    "TextEncoder",
    "SanaDenoiser",
    "load_model",
]

import torch.nn as nn

from torch import Tensor
from typing import Dict, Optional, Sequence, Tuple, Union

from azula.denoise import Gaussian, GaussianDenoiser
from azula.nn.utils import skip_init
from azula.noise import RectifiedSchedule, Schedule

from ..utils import as_dtype, load_cards


class AutoEncoder(nn.Module):
    r"""Creates an auto-encoder wrapper."""

    def __init__(
        self,
        ae: nn.Module,  # AutoencoderDC
        scale: float = 1.0,
    ):
        super().__init__()

        self.ae = ae
        self.scale = scale

    def encode(self, x: Tensor) -> Tensor:
        r"""Encodes images to latents.

        Arguments:
            x: A batch of images :math:`x`, with shape :math:`(B, 3, H, W)`.
                Pixel values are expected to range between -1 and 1.

        Returns:
            A batch of latents :math:`z \sim E(x)`, with shape :math:`(B, 32, H / 32, W / 32)`.
        """

        dtype = {"dtype": self.ae.dtype, "device": self.ae.device}

        z = self.ae.encode(x.to(**dtype)).latent
        z = z * self.scale

        return z.to(x)

    def decode(self, z: Tensor) -> Tensor:
        r"""Decodes latents to images.

        Arguments:
            z: A batch of latents :math:`z`, with shape :math:`(B, 32, H / 32, W / 32)`.

        Returns:
            A batch of images :math:`x = D(z)`, with shape :math:`(B, 3, H, W)`.
        """

        dtype = {"dtype": self.ae.dtype, "device": self.ae.device}

        z = z / self.scale
        x = self.ae.decode(z.to(**dtype)).sample

        return x.to(z)


class TextEncoder(nn.Module):
    r"""Creates a text encoder."""

    def __init__(
        self,
        gemma: nn.Module,
        tokenizer: nn.Module,
        max_length: int = 300,
    ):
        super().__init__()

        self.gemma = gemma

        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "right"

        self.max_length = max_length

    def forward(
        self,
        prompt: Union[str, Sequence[str]],
        instructions: Sequence[str] = [
            "Given a user prompt, generate an 'Enhanced prompt' that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:",
            "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.",
            "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.",
            "Here are examples of how to transform or refine prompts:",
            "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.",
            "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.",
            "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:",
            "User Prompt: ",
        ],
    ) -> Dict[str, Tensor]:
        r"""
        Arguments:
            prompt: A text prompt or list of text prompts.
            instructions: A set of human instructions to prepend to each prompt.

        Returns:
            The Gemma-encoded prompt(s).
        """

        if isinstance(prompt, str):
            prompt = [prompt]

        prompt = [text.lower().strip() for text in prompt]

        if instructions:
            chi = "\n".join(instructions)
            prompt = [chi + text if text else "" for text in prompt]
            max_length_all = self.max_length + len(self.tokenizer.encode(chi)) - 2
        else:
            max_length_all = self.max_length

        tokens = self.tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length_all,
            padding="max_length",
            return_tensors="pt",
        )

        prompt_mask = tokens.attention_mask.to(device=self.gemma.device)
        prompt_embeds = self.gemma(
            tokens.input_ids.to(device=self.gemma.device),
            attention_mask=prompt_mask,
            output_hidden_states=False,
        ).last_hidden_state

        select = [0, *range(-self.max_length + 1, 0)]

        return {
            "prompt_embeds": prompt_embeds[:, select],
            "prompt_mask": prompt_mask[:, select],
        }


class SanaDenoiser(GaussianDenoiser):
    r"""Creates a Sana denoiser.

    Arguments:
        backbone: A time conditional network.
        schedule: A noise schedule. If :py:`None`, use
            :class:`azula.noise.RectifiedSchedule` instead.
    """

    def __init__(
        self,
        backbone: nn.Module,
        schedule: Optional[Schedule] = None,
    ):
        super().__init__()

        self.backbone = backbone

        if schedule is None:
            self.schedule = RectifiedSchedule()
        else:
            self.schedule = schedule

    def forward(
        self,
        z_t: Tensor,
        t: Tensor,
        prompt_embeds: Tensor,
        prompt_mask: Tensor,
        **kwargs,
    ) -> Gaussian:
        r"""
        Arguments:
            z_t: A noisy tensor :math:`z_t`, with shape :math:`(B, C, H, W)`.
            t: The time :math:`t`, with shape :math:`()` or :math:`(B)`.
            prompt_embeds: The Gemma-encoded text prompt :math:`y`, with shape :math:`(B, L, D)`.
            prompt_mask: The text attention mask, with shape :math:`(B, L)`.
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
        c_time = 1000 * (sigma_t / (alpha_t + sigma_t)).flatten()
        c_var = sigma_t**2 / (alpha_t**2 + sigma_t**2)

        B, _, _, _ = z_t.shape
        _, L, D = prompt_embeds.shape

        dtype = {"dtype": self.backbone.dtype, "device": self.backbone.device}

        output = self.backbone(
            timestep=c_time.to(**dtype).expand(B),
            hidden_states=(c_in * z_t).to(**dtype),
            encoder_hidden_states=prompt_embeds.to(**dtype).expand(B, L, D),
            encoder_attention_mask=prompt_mask.to(**dtype).expand(B, L),
            **kwargs,
        ).sample

        mean = c_skip * z_t + c_out * output.to(z_t)
        var = c_var

        return Gaussian(mean=mean, var=var)


def load_model(
    name: str,
    **kwargs,
) -> Tuple[GaussianDenoiser, AutoEncoder, TextEncoder]:
    r"""Loads a pre-trained Sana latent denoiser.

    Arguments:
        name: The pre-trained model name.
        kwargs: Keyword arguments passed to :func:`diffusers.SanaPipeline.from_pretrained`.

    Returns:
        A pre-trained latent denoiser and the corresponding auto-encoder and text encoder.
    """

    from diffusers import SanaPipeline

    card = load_cards(__name__)[name]

    with skip_init():
        pipe = SanaPipeline.from_pretrained(
            card.repo,
            torch_dtype=as_dtype(card.dtype),
            variant=card.variant,
            **kwargs,
        )

    denoiser = SanaDenoiser(backbone=pipe.transformer)

    autoencoder = AutoEncoder(
        ae=pipe.vae,
        scale=pipe.vae.config.scaling_factor,
    )

    textencoder = TextEncoder(
        gemma=pipe.text_encoder,
        tokenizer=pipe.tokenizer,
    )

    return denoiser, autoencoder, textencoder
