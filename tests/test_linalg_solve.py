r"""Tests for the azula.linalg.solve module."""

import pytest
import torch

from functools import partial
from typing import Sequence

from azula.linalg.solve import cg, gmres


@pytest.fixture(autouse=True, scope="module")
def torch_float64():
    default = torch.get_default_dtype()

    try:
        yield torch.set_default_dtype(torch.float64)
    finally:
        torch.set_default_dtype(default)


@pytest.mark.parametrize("rank", [3, 5])
@pytest.mark.parametrize("batch", [(), (64,)])
def test_cg(rank: int, batch: Sequence[int]):
    U = torch.randn(*batch, 5, rank)
    A = partial(torch.einsum, "...ij,...j", U @ U.mT)

    x = torch.randn(*batch, 5)
    Ax = A(x)

    # x_0 = 0
    y = cg(A=A, b=Ax, iterations=rank)
    Ay = A(y)

    assert y.shape == x.shape
    assert torch.allclose(Ay, Ax, atol=1e-6)

    # x_0 = x
    y = cg(A=A, b=Ax, x0=x, iterations=1)
    Ay = A(y)

    assert y.shape == x.shape
    assert torch.allclose(Ay, Ax)


@pytest.mark.parametrize("rank", [3, 5])
@pytest.mark.parametrize("batch", [(), (64,)])
def test_gmres(rank: int, batch: Sequence[int]):
    U = torch.randn(*batch, 5, rank)
    V = torch.randn(*batch, rank, 5)
    A = partial(torch.einsum, "...ij,...j", U @ V)

    x = torch.randn(*batch, 5)
    Ax = A(x)

    # x_0 = 0
    y = gmres(A=A, b=Ax, iterations=rank)
    Ay = A(y)

    assert y.shape == x.shape
    assert torch.allclose(Ay, Ax, atol=1e-6)

    # x_0 = x
    y = gmres(A=A, b=Ax, x0=x, iterations=1)
    Ay = A(y)

    assert y.shape == x.shape
    assert torch.allclose(Ay, Ax)
