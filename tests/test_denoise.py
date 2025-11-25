r"""Tests for the azula.denoise module."""

import pytest
import torch
import torch.nn as nn

from torch import Tensor
from torch.distributions import Normal
from typing import Any, Sequence

from azula.denoise import GaussianDenoiser, GaussianPosterior, KarrasDenoiser, Posterior
from azula.linalg.covariance import DPLRCovariance, KroneckerCovariance
from azula.nn.embedding import SineEncoding
from azula.noise import VPSchedule


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


@pytest.mark.parametrize("with_label", [False, True])
@pytest.mark.parametrize("batch", [(), (64,)])
@pytest.mark.parametrize("channels", [5])
def test_KarrasDenoiser(with_label: bool, batch: Sequence[int], channels: int):
    denoiser = KarrasDenoiser(
        backbone=Dummy(channels, with_label),
        schedule=VPSchedule(),
    )

    # Forward
    x = torch.randn(*batch, channels)
    t = torch.rand(batch)

    if with_label:
        q = denoiser(x, t, label="cat")
    else:
        q = denoiser(x, t)

    assert isinstance(q, Posterior)
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
