r"""Tests for the azula.noise module."""

import pytest
import torch

from typing import Sequence

from azula.noise import VESchedule, VPSchedule


@pytest.mark.parametrize("batch", [(), (64,)])
def test_schedules(batch: Sequence[int]):
    Ss = [
        VPSchedule,
        VESchedule,
    ]

    for S in Ss:
        schedule = S()

        # Scales
        t = torch.rand(batch)
        alpha_t, sigma_t = schedule(t)

        assert alpha_t.shape == (*batch, 1), S
        assert sigma_t.shape == (*batch, 1), S

        assert torch.all(alpha_t > 0), S
        assert torch.all(sigma_t > 0), S

        # Monotonicity
        s = torch.rand_like(t) * t
        alpha_s, sigma_s = schedule(s)

        assert torch.all(alpha_s / sigma_s >= alpha_t / sigma_t), S

        # Start time
        alpha_0, _ = schedule(torch.zeros(()))

        assert torch.all(alpha_0 == 1), S
