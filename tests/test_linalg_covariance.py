r"""Tests for the azula.linalg.solve module."""

import pytest
import torch

from collections.abc import Iterator, Sequence

from azula.linalg.covariance import (
    DiagonalCovariance,
    DPLRCovariance,
    FullCovariance,
    IsotropicCovariance,
    KroneckerCovariance,
)


@pytest.fixture(autouse=True, scope="module")
def torch_float64() -> Iterator[None]:
    default = torch.get_default_dtype()

    try:
        yield torch.set_default_dtype(torch.float64)
    finally:
        torch.set_default_dtype(default)


@pytest.mark.parametrize(
    "covariance_cls",
    [
        IsotropicCovariance,
        DiagonalCovariance,
        FullCovariance,
        DPLRCovariance,
        KroneckerCovariance,
    ],
)
@pytest.mark.parametrize("shape", [(5,), (3, 5)])
@pytest.mark.parametrize("batch", [(), (256,), (16, 16)])
@pytest.mark.parametrize("rank", [2])
def test_covariances(
    covariance_cls: type,
    shape: Sequence[int],
    batch: Sequence[int],
    rank: int,
) -> None:
    X = torch.randn(1024, *shape)

    try:
        cov = covariance_cls.from_data(X, rank=rank)
    except TypeError:
        cov = covariance_cls.from_data(X)

    x = torch.randn(*batch, *shape)

    assert x.shape == cov(x).shape
    assert x.shape == cov.inv(x).shape

    assert torch.allclose(x, cov.inv(cov(x)))
    assert torch.allclose(x, cov(cov.inv(x)))

    assert torch.allclose(cov.logdet(), -cov.inv.logdet())
