r"""Tests for the azula.sample module."""

import pytest
import torch
import torch.nn as nn

from azula.denoise import PreconditionedDenoiser
from azula.nn.embedding import SineEncoding
from azula.noise import VPSchedule
from azula.sample import DDIMSampler, DDPMSampler
from functools import partial
from torch import Tensor
from typing import Any, Sequence


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
def test_samplers(with_label: bool, batch: Sequence[int]):
    denoiser = PreconditionedDenoiser(
        backbone=Dummy(5, with_label),
        schedule=VPSchedule(),
    )

    Ss = [
        partial(DDPMSampler, steps=64),
        partial(DDIMSampler, steps=64, eta=None),
        partial(DDIMSampler, steps=64, eta=1.0),
    ]

    for S in Ss:
        sampler = S(denoiser)

        z = torch.randn(*batch, 5)

        if with_label:
            x = sampler(z, label="cat")
        else:
            x = sampler(z)

        assert x.shape == z.shape
        assert torch.all(torch.isfinite(x))
