r"""Tests for the azula.linalg.solve module."""

import pytest
import torch

from typing import Sequence

from azula.linalg.covariance import (
    DiagonalCovariance,
    DPLRCovariance,
    FullCovariance,
    IsotropicCovariance,
    KroneckerCovariance,
)


@pytest.fixture(autouse=True, scope="module")
def torch_float64():
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
def test_covariances(covariance_cls: type, shape: Sequence[int], batch: Sequence[int]):
    X = torch.randn(1024, *shape)

    cov = covariance_cls.from_data(X)
    cov_inv = cov.inv

    x = torch.randn(*batch, *shape)

    assert x.shape == cov(x).shape
    assert x.shape == cov_inv(x).shape

    assert torch.allclose(x, cov_inv(cov(x)))
    assert torch.allclose(x, cov(cov_inv(x)))
