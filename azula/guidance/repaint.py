r"""RePaint internals.

References:
    | RePaint: Inpainting using Denoising Diffusion Probabilistic Models (Lugmayr et al., 2022)
    | https://arxiv.org/abs/2201.09865
"""

__all__ = [
    "RePaintSampler",
]

import torch

from torch import BoolTensor, Tensor

from ..denoise import Denoiser
from ..sample import DDIMSampler


class RePaintSampler(DDIMSampler):
    r"""Creates a RePaint sampler.

    Arguments:
        denoiser: A denoiser :math:`q_\phi(X \mid X_t)`.
        y: An observation :math:`y = m \odot x`.
        mask: The observation mask :math:`m`.
        iterations: The number of RePaint iterations per step.
        kwargs: Keyword arguments passed to :class:`azula.sample.DDIMSampler`.
    """

    def __init__(
        self,
        denoiser: Denoiser,
        y: Tensor,
        mask: BoolTensor,
        iterations: int = 3,
        **kwargs,
    ):
        super().__init__(denoiser, **kwargs)

        self.y = y
        self.mask = mask

        self.iterations = iterations

    @torch.no_grad()
    def step(self, x_t: Tensor, t: Tensor, s: Tensor, **kwargs) -> Tensor:
        alpha_s, sigma_s = self.denoiser.schedule(s)
        alpha_t, sigma_t = self.denoiser.schedule(t)

        for _ in range(self.iterations):
            x_s = super().step(x_t, t, s, **kwargs)
            x_s = torch.where(
                self.mask,
                alpha_s * self.y + sigma_s * torch.randn_like(self.y),
                x_s,
            )

            x_t = alpha_t / alpha_s * x_s + alpha_t * torch.sqrt(
                (sigma_t / alpha_t) ** 2 - (sigma_s / alpha_s) ** 2
            ) * torch.randn_like(x_s)

        return x_s
