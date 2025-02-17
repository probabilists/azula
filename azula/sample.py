r"""Reverse diffusion samplers.

For a distribution :math:`p(X)` over :math:`\mathbb{R}^D`, the perturbation kernel (see
:mod:`azula.noise`)

.. math:: p(X_t \mid X) = \mathcal{N}(X_t \mid \alpha_t X_t, \sigma_t^2 I)

defines a series of marginal distributions

.. math:: p(X_t) = \int p(X_t \mid x) \, p(x) \operatorname{d}\!x \, .

The goal of diffusion models is to generate samples from :math:`p(X_0)`. To this end,
reverse transition kernels :math:`q(X_s \mid X_t)` from :math:`t` to :math:`s < t` are
chosen. Then, starting from :math:`x_1 \sim p(X_1)`, :math:`T` transitions
:math:`x_{t_{i-1}} \sim q(X_{t_{i-1}} \mid x_{t_i})` are simulated from :math:`t_T = 1` to
:math:`t_0 = 0`. If the kernels are consistent with the marginals :math:`p(X_t)`, that
is if

.. math:: p(X_{t_{i-1}}) \approx
    \int q(X_{t_{i-1}} \mid x_{t_i}) \, p(x_{t_i}) \, \operatorname{d}\!x_{t_i} \, ,

for all :math:`i = 1, \dots, T`, the tensors :math:`x_{t_i}` are distributed according
to :math:`p(X_{t_i})`, including the last one :math:`x_{t_0} = x_0`.
"""

__all__ = [
    "Sampler",
    "DDPMSampler",
    "DDIMSampler",
    "EulerSampler",
    "HeunSampler",
    "LMSSampler",
    "PCSampler",
]

import torch
import torch.nn as nn

from torch import Tensor
from typing import Optional, Sequence
from zuko.utils import gauss_legendre

from .denoise import GaussianDenoiser


class Sampler(nn.Module):
    r"""Abstract reverse diffusion sampler.

    Arguments:
        steps: The number of discretization steps :math:`T`. By default, the step size
            :math:`t - s` is constant.
    """

    denoiser: GaussianDenoiser

    def __init__(self, steps: int):
        super().__init__()

        self.register_buffer("timesteps", torch.linspace(1, 0, steps + 1))

    @torch.no_grad()
    def init(
        self,
        shape: Sequence[int],
        mean: Optional[Tensor] = None,
        var: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        r"""Draws an initial noisy tensor :math:`x_1`.

        .. math:: x_1 \sim \mathcal{N}(\alpha_1 \mathbb{E}[X], \alpha_1^2 \mathbb{V}[X] + \sigma_1^2 I)

        Arguments:
            shape: The shape :math:`(*)` of the tensor.
            mean: The mean :math:`\mathbb{E}[X]` of :math:`p(X)`, with shape
                :math:`()` or :math:`(*)`. If :py:`None`, use 0 instead.
            var: The variance :math:`\mathbb{V}[X]` of :math:`p(X)`, with shape
                :math:`()` or :math:`(*)`. If :py:`None`, use 1 instead.
            kwargs: Keyword arguments passed to :func:`torch.randn`.

        Returns:
            A noisy tensor :math:`x_1`, with shape :math:`(*)`.
        """

        kwargs.setdefault("dtype", self.timesteps.dtype)
        kwargs.setdefault("device", self.timesteps.device)

        alpha_1, sigma_1 = self.denoiser.schedule(self.timesteps[0])

        if mean is None:
            mean = torch.zeros_like(alpha_1)

        if var is None:
            var = torch.ones_like(sigma_1)

        z = torch.randn(shape, **kwargs)

        return alpha_1 * mean + torch.sqrt(alpha_1**2 * var + sigma_1**2) * z

    @torch.no_grad()
    def forward(self, x_1: Tensor, **kwargs) -> Tensor:
        r"""Simulates the reverse process from :math:`t = 1` to :math:`0`.

        Arguments:
            x_1: A noisy tensor :math:`x_1`, with shape :math:`(*, D)`.
            kwargs: Optional keyword arguments.

        Returns:
            The clean tensor :math:`x_0`, with shape :math:`(*, D)`.
        """

        x_t = x_1

        for t, s in self.timesteps.unfold(0, 2, 1):
            x_s = self.step(x_t, t, s, **kwargs)
            x_t = x_s

        x_0 = x_t

        return x_0

    def step(self, x_t: Tensor, t: Tensor, s: Tensor, **kwargs) -> Tensor:
        r"""Simulates the reverse process from :math:`t` to :math:`s \leq t`.

        Arguments:
            x_t: The current tensor :math:`x_t`, with shape :math:`(*, D)`.
            t: The current time :math:`t`, with shape :math:`(*)`.
            s: The target time :math:`s`, with shape :math:`(*)`.
            kwargs: Optional keyword arguments.

        Returns:
            The new tensor :math:`x_s \sim q(X_s \mid x_t)`, with shape :math:`(*, D)`.
        """

        raise NotImplementedError()


class DDPMSampler(Sampler):
    r"""Creates an DDPM sampler.

    .. math:: x_s = \alpha_s \mu_\phi(x_t)
        + \sigma_s \, \sqrt{1 - \tau} \, \frac{x_t - \alpha_t \mu_\phi(x_t)}{\sigma_t}
        + \sigma_s \, \sqrt{\tau} \, \epsilon

    where :math:`\epsilon \sim \mathcal{N}(0, I)` and

    .. math:: \tau = 1 - \frac{\alpha_t^2}{\alpha_s^2} \frac{\sigma_s^2}{\sigma_t^2} \, .

    References:
        | Denoising Diffusion Probabilistic Models (Ho et al., 2020)
        | https://arxiv.org/abs/2006.11239

    Arguments:
        denoiser: A Gaussian denoiser.
        kwargs: Keyword arguments passed to :class:`Sampler`.
    """

    def __init__(self, denoiser: GaussianDenoiser, **kwargs):
        super().__init__(**kwargs)

        self.denoiser = denoiser

    def step(self, x_t: Tensor, t: Tensor, s: Tensor, **kwargs) -> Tensor:
        alpha_s, sigma_s = self.denoiser.schedule(s)
        alpha_t, sigma_t = self.denoiser.schedule(t)

        tau = 1 - (alpha_t / alpha_s * sigma_s / sigma_t) ** 2
        eps = torch.randn_like(x_t)

        q = self.denoiser(x_t, t, **kwargs)

        x_s = alpha_s * q.mean
        x_s = x_s + sigma_s * torch.sqrt(1 - tau) / sigma_t * (x_t - alpha_t * q.mean)
        x_s = x_s + sigma_s * torch.sqrt(tau) * eps

        return x_s


class DDIMSampler(Sampler):
    r"""Creates a DDIM sampler.

    .. math:: x_s = \alpha_s \mu_\phi(x_t)
        + \sigma_s \, \sqrt{1 - \eta \, \tau} \, \frac{x_t - \alpha_t \mu_\phi(x_t)}{\sigma_t}
        + \sigma_s \, \sqrt{\eta \, \tau} \, \epsilon

    where :math:`\epsilon \sim \mathcal{N}(0, I)` and

    .. math:: \tau = 1 - \frac{\alpha_t^2}{\alpha_s^2} \frac{\sigma_s^2}{\sigma_t^2} \, .

    References:
        | Denoising Diffusion Implicit Models (Song et al., 2021)
        | https://arxiv.org/abs/2010.02502

    Arguments:
        denoiser: A Gaussian denoiser.
        eta: The stochasticity hyperparameter :math:`\eta \in \mathbb{R}_+`.
            If :math:`\eta = 1`, :class:`DDIMSampler` is equivalent to :class:`DDPMSampler`.
            If :math:`\eta = 0`, :class:`DDIMSampler` is equivalent to :class:`EulerSampler`.
        kwargs: Keyword arguments passed to :class:`Sampler`.
    """

    def __init__(self, denoiser: GaussianDenoiser, eta: float = 0.0, **kwargs):
        super().__init__(**kwargs)

        self.denoiser = denoiser

        self.register_buffer("eta", torch.as_tensor(eta))

    def step(self, x_t: Tensor, t: Tensor, s: Tensor, **kwargs) -> Tensor:
        alpha_s, sigma_s = self.denoiser.schedule(s)
        alpha_t, sigma_t = self.denoiser.schedule(t)

        tau = 1 - (alpha_t / alpha_s * sigma_s / sigma_t) ** 2
        tau = torch.clip(self.eta * tau, min=0, max=1)
        eps = torch.randn_like(x_t)

        q = self.denoiser(x_t, t, **kwargs)

        x_s = alpha_s * q.mean
        x_s = x_s + sigma_s * torch.sqrt(1 - tau) / sigma_t * (x_t - alpha_t * q.mean)
        x_s = x_s + sigma_s * torch.sqrt(tau) * eps

        return x_s


class EulerSampler(Sampler):
    r"""Creates a deterministic Euler (1st order) sampler.

    The integration is carried with respect to the noise-to-signal ratio
    :math:`\frac{\sigma_t}{\alpha_t}` rather than the time :math:`t`.

    Wikipedia:
        https://wikipedia.org/wiki/Euler_method

    Arguments:
        denoiser: A Gaussian denoiser.
        kwargs: Keyword arguments passed to :class:`Sampler`.
    """

    def __init__(self, denoiser: GaussianDenoiser, **kwargs):
        super().__init__(**kwargs)

        self.denoiser = denoiser

    def step(self, x_t: Tensor, t: Tensor, s: Tensor, **kwargs) -> Tensor:
        alpha_s, sigma_s = self.denoiser.schedule(s)
        alpha_t, sigma_t = self.denoiser.schedule(t)

        q_t = self.denoiser(x_t, t, **kwargs)
        z_t = (x_t - alpha_t * q_t.mean) / sigma_t
        x_s = alpha_s / alpha_t * x_t + alpha_s * (sigma_s / alpha_s - sigma_t / alpha_t) * z_t

        return x_s


class HeunSampler(Sampler):
    r"""Creates a deterministic Heun (2nd order) sampler.

    The integration is carried with respect to the noise-to-signal ratio
    :math:`\frac{\sigma_t}{\alpha_t}` rather than the time :math:`t`.

    Wikipedia:
        https://wikipedia.org/wiki/Heun%27s_method

    Arguments:
        denoiser: A Gaussian denoiser.
        kwargs: Keyword arguments passed to :class:`Sampler`.
    """

    def __init__(self, denoiser: GaussianDenoiser, **kwargs):
        super().__init__(**kwargs)

        self.denoiser = denoiser

    def step(self, x_t: Tensor, t: Tensor, s: Tensor, **kwargs) -> Tensor:
        alpha_s, sigma_s = self.denoiser.schedule(s)
        alpha_t, sigma_t = self.denoiser.schedule(t)

        q_t = self.denoiser(x_t, t, **kwargs)
        z_t = (x_t - alpha_t * q_t.mean) / sigma_t
        x_s = alpha_s / alpha_t * x_t + alpha_s * (sigma_s / alpha_s - sigma_t / alpha_t) * z_t

        q_s = self.denoiser(x_s, s, **kwargs)
        z_s = (x_s - alpha_s * q_s.mean) / sigma_s
        z_t = (z_t + z_s) / 2
        x_s = alpha_s / alpha_t * x_t + alpha_s * (sigma_s / alpha_s - sigma_t / alpha_t) * z_t

        return x_s


class LMSSampler(Sampler):
    r"""Creates a linear multi-step (LMS) sampler.

    References:
        | k-diffusion (Katherine Crowson)
        | https://github.com/crowsonkb/k-diffusion

    Arguments:
        denoiser: A Gaussian denoiser.
        order: The order of the multi-step method.
        kwargs: Keyword arguments passed to :class:`Sampler`.
    """

    def __init__(self, denoiser: GaussianDenoiser, order: int = 3, **kwargs):
        super().__init__(**kwargs)

        self.denoiser = denoiser
        self.order = order

    @staticmethod
    def adams_bashforth(t: Tensor, i: int, order: int = 3) -> Tensor:
        r"""Returns the coefficients of the :math:`N`-th order Adams-Bashforth method.

        Wikipedia:
            https://wikipedia.org/wiki/Linear_multistep_method

        Arguments:
            t: The integration variable, with shape :math:`(T)`.
            i: The integration step.
            order: The method order :math:`N`.

        Returns:
            The Adams-Bashforth coefficients, with shape :math:`(N)`.
        """

        ti = t[i]
        tj = t[i - order : i]
        tk = torch.cat((tj, tj)).unfold(0, order, 1)[:order, 1:]

        tj_tk = tj[..., None] - tk

        # Lagrange basis
        def lj(t):
            return torch.prod((t[..., None, None] - tk) / tj_tk, dim=-1)

        # Adams-Bashforth coefficients
        cj = gauss_legendre(lj, tj[-1], ti, n=order // 2 + 1)

        return cj

    @torch.no_grad()
    def forward(self, x_1: Tensor, **kwargs) -> Tensor:
        alpha, sigma = self.denoiser.schedule(self.timesteps)
        ratio = sigma.double() / alpha.double()

        x_t = x_1

        derivatives = []

        for i, t in enumerate(self.timesteps[:-1]):
            alpha_t, sigma_t = alpha[i], sigma[i]
            alpha_s = alpha[i + 1]

            q_t = self.denoiser(x_t, t, **kwargs)
            z_t = (x_t - alpha_t * q_t.mean) / sigma_t

            derivatives.append(z_t)

            if len(derivatives) > self.order:
                derivatives.pop(0)

            coefficients = self.adams_bashforth(ratio, i + 1, order=len(derivatives))
            coefficients = coefficients.to(x_t)

            delta = sum(c * d for c, d in zip(coefficients, derivatives))

            x_t = alpha_s * (x_t / alpha_t + delta)

        x_0 = x_t

        return x_0


class PCSampler(Sampler):
    r"""Creates a predictor-corrector (PC) sampler.

    References:
        | Score-Based Generative Modeling through Stochastic Differential Equations (Song et al., 2021)
        | https://arxiv.org/abs/2011.13456

    Arguments:
        denoiser: A Gaussian denoiser.
        corrections: The number of Langevin corrections per step.
        delta: The amplitude of Langevin corrections.
        kwargs: Keyword arguments passed to :class:`Sampler`.
    """

    def __init__(
        self,
        denoiser: GaussianDenoiser,
        corrections: int = 1,
        delta: float = 0.01,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.denoiser = denoiser
        self.corrections = corrections

        self.register_buffer("delta", torch.as_tensor(delta))

    def step(self, x_t: Tensor, t: Tensor, s: Tensor, **kwargs) -> Tensor:
        alpha_s, sigma_s = self.denoiser.schedule(s)
        alpha_t, sigma_t = self.denoiser.schedule(t)

        # Corrector
        for _ in range(self.corrections):
            q = self.denoiser(x_t, t, **kwargs)
            x_t = (
                x_t
                + self.delta * (alpha_t * q.mean - x_t)
                + torch.sqrt(2 * self.delta) * sigma_t * torch.randn_like(x_t)
            )

        # Predictor
        q = self.denoiser(x_t, t, **kwargs)
        x_s = alpha_s * q.mean + sigma_s / sigma_t * (x_t - alpha_t * q.mean)

        return x_s
