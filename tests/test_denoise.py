r"""Tests for the azula.denoise module."""

import pytest
import torch
import torch.nn as nn

from torch import Tensor
from torch.distributions import Normal
from typing import Any, Sequence

from azula.denoise import Gaussian, PreconditionedDenoiser
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
def test_Gaussian(isotropic: bool, batch: Sequence[int]):
    mean = torch.randn(*batch, 5)

    if isotropic:
        std = torch.rand(*batch, 1) + 1e-3
    else:
        std = torch.rand(*batch, 5) + 1e-3

    x = torch.normal(mean, std)

    log_q = Gaussian(mean, std**2).log_prob(x)
    log_p = Normal(mean, std).log_prob(x).sum(dim=-1)

    assert log_q.shape == batch
    assert torch.allclose(log_q, log_p, atol=1e-6)


@pytest.mark.parametrize("with_label", [False, True])
@pytest.mark.parametrize("batch", [(), (64,)])
def test_PreconditionedDenoiser(with_label: bool, batch: Sequence[int]):
    denoiser = PreconditionedDenoiser(
        backbone=Dummy(5, with_label),
        schedule=VPSchedule(),
    )

    # Forward
    x = torch.randn(*batch, 5)
    t = torch.rand(batch)

    if with_label:
        q = denoiser(x, t, label="cat")
    else:
        q = denoiser(x, t)

    assert isinstance(q, Gaussian)
    assert q.mean.shape == x.shape
    assert q.var.expand(x.shape).shape == x.shape

    # Loss
    if with_label:
        loss = denoiser.loss(x, t, label="cat")
    else:
        loss = denoiser.loss(x, t)

    assert loss.shape == batch
    assert loss.requires_grad

    loss.mean().backward()

    for p in denoiser.parameters():
        assert p.grad is not None
        assert torch.all(torch.isfinite(p.grad))
