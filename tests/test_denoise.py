r"""Tests for the azula.denoise module."""

import pytest
import torch
import torch.nn as nn

from torch import Tensor
from torch.distributions import Normal
from typing import Any, Sequence
from typing import Any, Sequence, Tuple

from azula.denoise import (
    GaussianDenoiser,
    GaussianPosterior,
    KarrasDenoiser,
    SimpleDenoiser,
    JiTDenoiser,
    Posterior,
    DiracPosterior,
)

from azula.linalg.covariance import DPLRCovariance, KroneckerCovariance
from azula.nn.embedding import SineEncoding
from azula.noise import RectifiedSchedule, Schedule, VPSchedule


class Dummy(nn.Module):
    def __init__(self, features: int = 5, with_label: bool = False):
        super().__init__()

        self.with_label = with_label

        self.l1 = nn.Linear(features, 64)
        self.l2 = nn.Linear(64, features)
        self.relu = nn.ReLU()

        self.time_encoding = SineEncoding(64)

    def forward(self, x_t: Tensor, t: Tensor, label: Any = None):
        y = self.l1(x_t)
        y = y + self.time_encoding(t)
        y = self.relu(y)
        y = self.l2(y)

        if self.with_label:
            assert label is not None
        else:
            assert label is None

        return y


@pytest.mark.parametrize("isotropic", [False, True])
@pytest.mark.parametrize("batch", [(), (64,)])
@pytest.mark.parametrize("channels", [5])
def test_GaussianPosterior(isotropic: bool, batch: Sequence[int], channels: int):
    mean = torch.randn(*batch, channels)

    if isotropic:
        std = torch.rand(*batch, 1) + 1e-3
    else:
        std = torch.rand(*batch, channels) + 1e-3

    x = mean + std * torch.randn_like(mean)

    log_q = GaussianPosterior(mean, std**2).log_prob(x)
    log_p = Normal(mean, std).log_prob(x)

    assert log_q.shape == (*batch, channels)
    assert torch.allclose(log_q, log_p, atol=1e-6)


@pytest.mark.parametrize("cov", ["dplr", "precond"])
@pytest.mark.parametrize("batch", [(), (64,)])
@pytest.mark.parametrize("channels", [5])
def test_GaussianDenoiser(cov: str, batch: Sequence[int], channels: int):
    data = torch.randn(256, channels)
    mean = torch.mean(data, dim=0)

    if cov == "dplr":
        cov = DPLRCovariance.from_data(data, rank=3)
    elif cov == "precond":
        cov = KroneckerCovariance.from_data(data, rank=0)

    denoiser = GaussianDenoiser(mean, cov, schedule=VPSchedule())

    # Forward
    x = torch.randn(*batch, channels, requires_grad=True)
    t = torch.rand(())

    q = denoiser(x, t)

    assert isinstance(q, Posterior)
    assert q.mean.shape == x.shape


class ReSchedule(Schedule):
    def __init__(self, schedule: Schedule):
        self.schedule = schedule

    def __call__(self, t: Tensor) -> Tuple[Tensor, Tensor]:
        alpha, sigma = self.schedule(t)
        return torch.ones_like(alpha), sigma / alpha


@pytest.mark.parametrize("denoiser_cls", [SimpleDenoiser, KarrasDenoiser])
@pytest.mark.parametrize("schedule_cls", [VPSchedule, RectifiedSchedule])
@pytest.mark.parametrize("with_label", [False, True])
@pytest.mark.parametrize("batch", [(), (64,)])
@pytest.mark.parametrize("channels", [5])
def test_denoisers(
    denoiser_cls: type,
    schedule_cls: type,
    with_label: bool,
    batch: Sequence[int],
    channels: int,
):
    denoiser = denoiser_cls(
        backbone=Dummy(channels, with_label),
        schedule=schedule_cls(),
    )

    # Forward
    x = torch.randn(*batch, channels)
    t = torch.rand(batch)

    alpha_t, sigma_t = denoiser.schedule(t)
    alpha_t, sigma_t = alpha_t[..., None], sigma_t[..., None]

    x_t = torch.normal(alpha_t * x, sigma_t)

    if with_label:
        q = denoiser(x_t, t, label="cat")
    else:
        q = denoiser(x_t, t)

    assert isinstance(q, Posterior)
    assert q.mean.shape == x.shape

    ## Reschedule to VE
    denoiser.schedule = ReSchedule(denoiser.schedule)

    if with_label:
        q_ve = denoiser(x_t / alpha_t, t, label="cat")
    else:
        q_ve = denoiser(x_t / alpha_t, t)

    assert torch.allclose(q.mean, q_ve.mean, atol=1e-6)

    # Loss
    if with_label:
        loss = denoiser.loss(x, t, label="cat")
    else:
        loss = denoiser.loss(x, t)

    assert loss.shape == ()
    assert loss.requires_grad

    loss.mean().backward()

    for p in denoiser.parameters():
        assert p.grad is not None
        assert torch.all(torch.isfinite(p.grad))


@pytest.mark.parametrize("with_label", [False, True])
@pytest.mark.parametrize("batch", [(), (64,)])
@pytest.mark.parametrize("channels", [5])
def test_JiTDenoiser(with_label: bool, batch: Sequence[int], channels: int):
    class JitSchedule(nn.Module):
        def forward(self, t: Tensor):
            return t, 1 - t

    denoiser = JiTDenoiser(
        backbone=Dummy(channels, with_label),
        schedule=JitSchedule(),
    )

    # Forward
    x = torch.randn(*batch, channels)
    t = torch.sigmoid(torch.randn(batch) * 0.8 + 0.8)

    if with_label:
        q = denoiser(x, t, label="cat")
    else:
        q = denoiser(x, t)

    assert isinstance(q, DiracPosterior)
    assert q.mean.shape == x.shape

    # Loss
    if with_label:
        loss = denoiser.loss(x, t, label="cat")
    else:
        loss = denoiser.loss(x, t)

    assert loss.shape == ()
    assert loss.requires_grad

    loss.mean().backward()

    for p in denoiser.parameters():
        assert p.grad is not None
        assert torch.all(torch.isfinite(p.grad))
