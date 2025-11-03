r"""Reverse diffusion samplers.

For a distribution :math:`p(X)` over :math:`\mathbb{R}^D`, the perturbation kernel (see
:mod:`azula.noise`)

.. math:: p(X_t \mid X) = \mathcal{N}(X_t \mid \alpha_t X_t, \sigma_t^2 I)

defines a series of marginal distributions

.. math:: p(X_t) = \int p(X_t \mid x) \, p(x) \, dx \, .

The goal of diffusion models is to generate samples from :math:`p(X_0)`. To this end,
reverse transition kernels :math:`q(X_s \mid X_t)` from :math:`t` to :math:`s < t` are
chosen. Then, starting from :math:`x_1 \sim p(X_1)`, :math:`T` transitions
:math:`x_{t_{i-1}} \sim q(X_{t_{i-1}} \mid x_{t_i})` are simulated from :math:`t_T = 1`
to :math:`t_0 = 0`. If the kernels are consistent with the marginals :math:`p(X_t)`,
that is if

.. math:: p(X_{t_{i-1}}) \approx
    \int q(X_{t_{i-1}} \mid x_{t_i}) \, p(x_{t_i}) \, dx_{t_i} \, ,

for all :math:`i = 1, \dots, T`, the tensors :math:`x_{t_i}` are distributed according
to :math:`p(X_{t_i})`, including the last one :math:`x_{t_0} = x_0`.
"""

__all__ = [
    "Sampler",
    "DDPMSampler",
    "DDIMSampler",
    "EulerSampler",
    "HeunSampler",
    "ABSampler",
    "EABSampler",
    "PCSampler",
]

import abc
import math
import torch

from torch import Tensor
from typing import Optional, Sequence, Union

from .denoise import Denoiser


class Sampler(abc.ABC):
    r"""Abstract reverse diffusion sampler.

    Arguments:
        start: The starting time :math:`t_T`.
        stop: The stopping time :math:`t_0`.
        steps: The number of discretization steps :math:`T`. By default, the step size
            :math:`t_{i} - t_{i-1}` is constant.
        dtype: The time data type.
        device: The time device.
    """

    denoiser: Denoiser

    def __init__(
        self,
        start: float = 1.0,
        stop: float = 0.0,
        steps: int = 64,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        self.start = start
        self.stop = stop
        self.steps = steps

        self.dtype = dtype
        self.device = device

    @property
    def timesteps(self) -> Tensor:
        return torch.linspace(
            self.start,
            self.stop,
            self.steps + 1,
            dtype=self.dtype,
            device=self.device,
        )

    @torch.no_grad()
    def init(
        self,
        shape: Sequence[int],
        mean: Union[float, Tensor] = 0.0,
        var: Union[float, Tensor] = 1.0,
        **kwargs,
    ) -> Tensor:
        r"""Draws an initial noisy tensor :math:`x_{t_T}`.

        .. math:: x_{t_T} \sim \mathcal{N}(\alpha_{t_T} \mathbb{E}[X], \alpha_{t_T}^2 \mathbb{V}[X] + \sigma_{t_T}^2 I)

        Arguments:
            shape: The shape :math:`(*)` of the tensor.
            mean: The mean :math:`\mathbb{E}[X]` of :math:`p(X)`, with shape
                :math:`()` or :math:`(*)`.
            var: The variance :math:`\mathbb{V}[X]` of :math:`p(X)`, with shape
                :math:`()` or :math:`(*)`.
            kwargs: Keyword arguments passed to :func:`torch.Tensor.to`.

        Returns:
            A noisy tensor :math:`x_{t_T}`, with shape :math:`(*)`.
        """

        t_T = self.timesteps[-1]

        alpha_T, sigma_T = self.denoiser.schedule(t_T)
        alpha_T, sigma_T = alpha_T.to(**kwargs), sigma_T.to(**kwargs)

        mean_T, std_T = alpha_T * mean, torch.sqrt(alpha_T**2 * var + sigma_T**2)
        mean_T, std_T = mean_T.expand(shape), std_T.expand(shape)

        return mean_T + std_T * torch.randn_like(mean_T)

    @torch.no_grad()
    def __call__(self, x: Tensor, **kwargs) -> Tensor:
        r"""Simulates the reverse process from :math:`t_T` to :math:`t_0`.

        Arguments:
            x: A noisy tensor :math:`x_{t_T}`, with shape :math:`(*)`.
            kwargs: Optional keyword arguments.

        Returns:
            The clean(er) tensor :math:`x_{t_0}`, with shape :math:`(*)`.
        """

        x_t = x

        for t, s in self.timesteps.unfold(0, 2, 1).to(device=x.device).unbind():
            x_s = self.step(x_t, t, s, **kwargs)
            x_t = x_s

        x = x_t

        return x

    def step(self, x_t: Tensor, t: Tensor, s: Tensor, **kwargs) -> Tensor:
        r"""Simulates the reverse process from :math:`t` to :math:`s < t`.

        Arguments:
            x_t: The current tensor :math:`x_t`, with shape :math:`(*)`.
            t: The current time :math:`t`, with shape :math:`()`.
            s: The target time :math:`s`, with shape :math:`()`.
            kwargs: Optional keyword arguments.

        Returns:
            The new tensor :math:`x_s \sim q(X_s \mid x_t)`, with shape :math:`(*)`.
        """

        raise NotImplementedError()


class DDPMSampler(Sampler):
    r"""Creates an DDPM sampler.

    .. math:: x_s \gets \alpha_s \mu_\phi(x_t)
        + \sigma_s \, \sqrt{1 - \tau} \, \frac{x_t - \alpha_t \mu_\phi(x_t)}{\sigma_t}
        + \sigma_s \, \sqrt{\tau} \, \epsilon

    where :math:`\epsilon \sim \mathcal{N}(0, I)` and

    .. math:: \tau = 1 - \frac{\alpha_t^2}{\alpha_s^2} \frac{\sigma_s^2}{\sigma_t^2} \, .

    References:
        | Denoising Diffusion Probabilistic Models (Ho et al., 2020)
        | https://arxiv.org/abs/2006.11239

    Arguments:
        denoiser: A denoiser :math:`q_\phi(X \mid X_t)`.
        kwargs: Keyword arguments passed to :class:`Sampler`.
    """

    def __init__(self, denoiser: Denoiser, **kwargs):
        super().__init__(**kwargs)

        self.denoiser = denoiser

    def step(self, x_t: Tensor, t: Tensor, s: Tensor, **kwargs) -> Tensor:
        alpha_s, sigma_s = self.denoiser.schedule(s)
        alpha_t, sigma_t = self.denoiser.schedule(t)

        tau = 1 - (alpha_t / alpha_s * sigma_s / sigma_t) ** 2
        eps = torch.randn_like(x_t)

        q_t = self.denoiser(x_t, t, **kwargs)

        x_s = alpha_s * q_t.mean
        x_s = x_s + sigma_s * torch.sqrt(1 - tau) / sigma_t * (x_t - alpha_t * q_t.mean)
        x_s = x_s + sigma_s * torch.sqrt(tau) * eps

        return x_s


class DDIMSampler(Sampler):
    r"""Creates a DDIM sampler.

    .. math:: x_s \gets \alpha_s \mu_\phi(x_t)
        + \sigma_s \, \sqrt{1 - \eta \, \tau} \, \frac{x_t - \alpha_t \mu_\phi(x_t)}{\sigma_t}
        + \sigma_s \, \sqrt{\eta \, \tau} \, \epsilon

    where :math:`\epsilon \sim \mathcal{N}(0, I)` and

    .. math:: \tau = 1 - \frac{\alpha_t^2}{\alpha_s^2} \frac{\sigma_s^2}{\sigma_t^2} \, .

    References:
        | Denoising Diffusion Implicit Models (Song et al., 2021)
        | https://arxiv.org/abs/2010.02502

    Arguments:
        denoiser: A denoiser :math:`q_\phi(X \mid X_t)`.
        eta: The stochasticity hyperparameter :math:`\eta \in \mathbb{R}_+`.
            If :math:`\eta = 1`, :class:`DDIMSampler` is equivalent to :class:`DDPMSampler`.
            If :math:`\eta = 0`, :class:`DDIMSampler` is equivalent to :class:`EulerSampler`.
        kwargs: Keyword arguments passed to :class:`Sampler`.
    """

    def __init__(self, denoiser: Denoiser, eta: float = 0.0, **kwargs):
        super().__init__(**kwargs)

        self.denoiser = denoiser
        self.eta = eta

    def step(self, x_t: Tensor, t: Tensor, s: Tensor, **kwargs) -> Tensor:
        alpha_s, sigma_s = self.denoiser.schedule(s)
        alpha_t, sigma_t = self.denoiser.schedule(t)

        tau = 1 - (alpha_t / alpha_s * sigma_s / sigma_t) ** 2
        tau = torch.clip(self.eta * tau, min=0, max=1)
        eps = torch.randn_like(x_t)

        q_t = self.denoiser(x_t, t, **kwargs)

        x_s = alpha_s * q_t.mean
        x_s = x_s + sigma_s * torch.sqrt(1 - tau) / sigma_t * (x_t - alpha_t * q_t.mean)
        x_s = x_s + sigma_s * torch.sqrt(tau) * eps

        return x_s


class EulerSampler(Sampler):
    r"""Creates a explicit Euler (1st order) sampler.

    Without loss of generality, let's assume :math:`\alpha_t = 1` and :math:`\sigma_t =
    t` such that

    .. math:: x_s = x_t + \int_t^s z(x_u) \, du

    where :math:`z(x_t) = \frac{x_t - \mathbb{E}[X \mid x_t]}{t}`. The explicit Euler
    step for this integral is

    .. math:: x_s \gets x_t + (s - t) \, z(x_t)

    Wikipedia:
        https://wikipedia.org/wiki/Euler_method

    Arguments:
        denoiser: A denoiser :math:`q_\phi(X \mid X_t)`.
        kwargs: Keyword arguments passed to :class:`Sampler`.
    """

    def __init__(self, denoiser: Denoiser, **kwargs):
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
    r"""Creates a explicit Heun (2nd order) sampler.

    Without loss of generality, let's assume :math:`\alpha_t = 1` and :math:`\sigma_t =
    t` such that

    .. math:: x_s = x_t + \int_t^s z(x_u) \, du

    where :math:`z(x_t) = \frac{x_t - \mathbb{E}[X \mid x_t]}{t}`. The explicit Heun
    step for this integral is

    .. math::
        x_s & \gets x_t + (s - t) \, z(x_t) \\
        x_s & \gets x_t + (s - t) \frac{z(x_t) + z(x_s)}{2}

    Wikipedia:
        https://wikipedia.org/wiki/Heun%27s_method

    Arguments:
        denoiser: A denoiser :math:`q_\phi(X \mid X_t)`.
        kwargs: Keyword arguments passed to :class:`Sampler`.
    """

    def __init__(self, denoiser: Denoiser, **kwargs):
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


class ABSampler(Sampler):
    r"""Creates an Adams-Bashforth (AB) multi-step sampler.

    Note:
        This sampler is equivalent to the :math:`\rho\mathrm{AB}` sampler from Zhang et al.
        (2023) and the linear multi-step (LMS) sampler from Katherine Crowson's
        `k-diffusion <https://github.com/crowsonkb/k-diffusion>`_.

    Without loss of generality, let's assume :math:`\alpha_t = 1` and :math:`\sigma_t =
    t` such that

    .. math:: x_s = x_t + \int_t^s z(x_u) \, du

    where :math:`z(x_t) = \frac{x_t - \mathbb{E}[X \mid x_t]}{t}`. The :math:`n`-th
    order Adams-Bashforth step for this integral is

    .. math:: x_s \gets x_t + \sum_{i=0}^{n-1} z(x_{t_i}) \int_t^s \ell_i(u) \, du

    where :math:`t_i` are previous time steps and the polynomials :math:`\ell_i(t)` form
    their Lagrange basis.

    .. math:: \ell_i(t_j) = \sum_{k=0}^{n-1} a_{ik} \, t_j^k = \delta_{ij}

    Therefore, the Adams-Bashforth coefficients are

    .. math:: \int_t^s \ell_i(u) \, du
        & = \sum_{k=0}^{n-1} a_{ik} \int_t^s u^k \, du \\
        & = \sum_{k=0}^{n-1} a_{ik} \left[ \frac{u^{k+1}}{k+1} \right]_t^s

    Wikipedia:
        https://wikipedia.org/wiki/Linear_multistep_method

    References:
        | Fast Sampling of Diffusion Models with Exponential Integrator (Zhang et al., 2023)
        | https://arxiv.org/abs/2204.13902

    Arguments:
        denoiser: A denoiser :math:`q_\phi(X \mid X_t)`.
        order: The order :math:`n` of the multi-step method.
        kwargs: Keyword arguments passed to :class:`Sampler`.
    """

    def __init__(self, denoiser: Denoiser, order: int = 3, **kwargs):
        super().__init__(**kwargs)

        self.denoiser = denoiser
        self.order = order

    @staticmethod
    def adams_bashforth(t: Tensor, n: int = 3) -> Tensor:
        r"""Returns the coefficients of the :math:`n`-th order Adams-Bashforth method.

        Arguments:
            t: The integration variable, with shape :math:`(m + 1)`.
            n: The method order :math:`n \leq m`.

        Returns:
            The coefficients, with shape :math:`(n)`.
        """

        m = len(t) - 1
        n = min(n, m)
        k = torch.arange(n, dtype=torch.float64, device=t.device)

        # Vandermonde matrix t_i^k
        V = t[m - n : m] ** k[:, None]

        # Integral of u^k from t_{m-1} to t_m
        b = t[m] ** (k + 1) / (k + 1) - t[m - 1] ** (k + 1) / (k + 1)

        return torch.linalg.solve(V, b).to(dtype=t.dtype)

    @torch.no_grad()
    def __call__(self, x: Tensor, **kwargs) -> Tensor:
        time = self.timesteps.to(device=x.device)
        alpha, sigma = self.denoiser.schedule(time)
        rho = sigma / alpha

        x_t = x

        buffer = []

        for i, t in enumerate(time[:-1].unbind()):
            alpha_t, sigma_t = alpha[i], sigma[i]
            alpha_s = alpha[i + 1]

            q_t = self.denoiser(x_t, t, **kwargs)
            z_t = (x_t - alpha_t * q_t.mean) / sigma_t

            buffer.append(z_t)
            if len(buffer) > self.order:
                buffer.pop(0)

            coeffs = self.adams_bashforth(rho[: i + 2], n=self.order)
            integral = sum(b * c for b, c in zip(buffer, coeffs))

            x_s = alpha_s / alpha_t * x_t + alpha_s * integral
            x_t = x_s

        x = x_t

        return x


class EABSampler(Sampler):
    r"""Creates an exponential Adams-Bashforth (EAB) multi-step sampler.

    Note:
        This sampler is a multi-step generalization of the DPM-Solver sampler from Cheng
        Lu's `dpm-solver <https://github.com/LuChengTHU/dpm-solver>`_.

    Without loss of generality, let's assume :math:`\alpha_t = 1` and :math:`\sigma_t =
    e^t` such that

    .. math:: x_s = x_t + \int_t^s e^u \, z(x_u) \, du

    where :math:`z(x_t) = \frac{x_t - \mathbb{E}[X \mid x_t]}{e^t}`. The :math:`n`-th
    order exponential Adams-Bashforth step for this integral is

    .. math:: x_s \gets x_t + \sum_{i=0}^{n-1} z(x_{t_i}) \int_t^s e^u \, \ell_i(u) \, du

    where :math:`t_i` are previous time steps and the polynomials :math:`\ell_i(t)` form
    their Lagrange basis.

    .. math:: \ell_i(t_j) = \sum_{k=0}^{n-1} a_{ik} \, t_j^k = \delta_{ij}

    Therefore, the exponential Adams-Bashforth coefficients are

    .. math:: \int_t^s e^u \, \ell_i(u) \, du
        & = \sum_{k=0}^{n-1} a_{ik} \int_t^s e^u \, u^k \, du \\
        & = \sum_{k=0}^{n-1} a_{ik} \left[ (-1)^k \, k! \, e^u \sum_{j=0}^k \frac{(-u)^j}{j!} \right]_t^s

    References:
        | Exponential Adams Bashforth ODE solver for stiff problems (Coudière et al., 2018)
        | https://arxiv.org/abs/1804.09927

        | DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps (Lu et al., 2022)
        | https://arxiv.org/abs/2206.00927

    Arguments:
        denoiser: A denoiser :math:`q_\phi(X \mid X_t)`.
        order: The order :math:`n` of the multi-step method.
        kwargs: Keyword arguments passed to :class:`Sampler`.
    """

    def __init__(self, denoiser: Denoiser, order: int = 3, **kwargs):
        super().__init__(**kwargs)

        self.denoiser = denoiser
        self.order = order

    @staticmethod
    def exponential_adams_bashforth(t: Tensor, n: int = 3) -> Tensor:
        r"""Returns the coefficients of the :math:`n`-th order exponential Adams-Bashforth method.

        Arguments:
            t: The integration variable, with shape :math:`(m + 1)`.
            n: The method order :math:`n \leq m`.

        Returns:
            The coefficients, with shape :math:`(n)`.
        """

        m = len(t) - 1
        n = min(n, m)
        k = torch.arange(n, dtype=torch.float64, device=t.device)
        k_fact = torch.lgamma(k + 1).exp()

        # Vandermonde matrix t_i^k
        V = t[m - n : m] ** k[:, None]

        # Integral of exp(u) u^k from t_{m-1} to t_m
        b = (
            (-1) ** k
            * k_fact
            * (
                torch.exp(t[m]) * torch.cumsum((-t[m]) ** k / k_fact, dim=0)
                - torch.exp(t[m - 1]) * torch.cumsum((-t[m - 1]) ** k / k_fact, dim=0)
            )
        )

        return torch.linalg.solve(V, b).to(dtype=t.dtype)

    @torch.no_grad()
    def __call__(self, x: Tensor, **kwargs) -> Tensor:
        time = self.timesteps.to(device=x.device)
        alpha, sigma = self.denoiser.schedule(time)
        log_rho = sigma.log() - alpha.log()

        x_t = x

        buffer = []

        for i, t in enumerate(time[:-1].unbind()):
            alpha_t, sigma_t = alpha[i], sigma[i]
            alpha_s = alpha[i + 1]

            q_t = self.denoiser(x_t, t, **kwargs)
            z_t = (x_t - alpha_t * q_t.mean) / sigma_t

            buffer.append(z_t)
            if len(buffer) > self.order:
                buffer.pop(0)

            coeffs = self.exponential_adams_bashforth(log_rho[: i + 2], n=self.order)
            integral = sum(b * c for b, c in zip(buffer, coeffs))

            x_s = alpha_s / alpha_t * x_t + alpha_s * integral
            x_t = x_s

        x = x_t

        return x


class PCSampler(Sampler):
    r"""Creates a predictor-corrector (PC) sampler.

    Arguments:
        denoiser: A denoiser :math:`q_\phi(X \mid X_t)`.
        corrections: The number of corrector steps for each predictor step.
        delta: The amplitude of corrector steps :math:`\delta \in [0,1]`.
        kwargs: Keyword arguments passed to :class:`Sampler`.
    """

    def __init__(
        self,
        denoiser: Denoiser,
        corrections: int = 1,
        delta: float = 0.01,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.denoiser = denoiser
        self.corrections = corrections
        self.delta = delta

    def step(self, x_t: Tensor, t: Tensor, s: Tensor, **kwargs) -> Tensor:
        alpha_s, sigma_s = self.denoiser.schedule(s)
        alpha_t, sigma_t = self.denoiser.schedule(t)

        # Corrector
        for _ in range(self.corrections):
            q_t = self.denoiser(x_t, t, **kwargs)
            x_t = (
                alpha_t * q_t.mean
                + math.sqrt(1 - self.delta) * (x_t - alpha_t * q_t.mean)
                + math.sqrt(self.delta) * sigma_t * torch.randn_like(x_t)
            )

        # Predictor
        q_t = self.denoiser(x_t, t, **kwargs)
        x_s = alpha_s * q_t.mean + sigma_s / sigma_t * (x_t - alpha_t * q_t.mean)

        return x_s
