r"""Tests for the azula.sample module."""

import pytest
import torch
import torch.nn as nn

from functools import partial
from torch import Tensor
from typing import Any, Sequence

from azula.denoise import KarrasDenoiser
from azula.nn.embedding import SineEncoding
from azula.noise import VPSchedule
from azula.sample import (
    DDIMSampler,
    DDPMSampler,
    EulerSampler,
    HeunSampler,
    PCSampler,
    REABSampler,
    vABSampler,
    xEABSampler,
    zABSampler,
    zEABSampler,
)


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


@pytest.mark.parametrize("with_label", [False, True])
@pytest.mark.parametrize("batch", [(), (64,)])
@pytest.mark.parametrize("channels", [5])
def test_samplers(with_label: bool, batch: Sequence[int], channels: int):
    denoiser = KarrasDenoiser(
        backbone=Dummy(channels, with_label),
        schedule=VPSchedule(),
    )

    Ss = [
        partial(DDPMSampler),
        partial(DDIMSampler, eta=0.0),
        partial(DDIMSampler, eta=1.0),
        partial(EulerSampler),
        partial(HeunSampler),
        partial(zABSampler),
        partial(vABSampler),
        partial(zEABSampler),
        partial(xEABSampler),
        partial(REABSampler),
        partial(PCSampler, corrections=1),
    ]

    for S in Ss:
        sampler = S(denoiser, steps=64)

        x1 = sampler.init((*batch, channels))

        assert x1.shape == (*batch, channels), S
        assert torch.all(torch.isfinite(x1)), S

        if with_label:
            x0 = sampler(x1, label="cat")
        else:
            x0 = sampler(x1)

        assert x0.shape == (*batch, channels), S
        assert torch.all(torch.isfinite(x0)), S
