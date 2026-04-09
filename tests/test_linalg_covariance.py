r"""Tests for the azula.linalg.solve module."""

import math
import pytest
import torch

from collections.abc import Callable, Iterator, Sequence
from functools import partial
from torch import Tensor

from azula.linalg.covariance import (
    Covariance,
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
    "covariance_fn",
    [
        IsotropicCovariance.from_data,
        DiagonalCovariance.from_data,
        FullCovariance.from_data,
        partial(DPLRCovariance.from_data, rank=1),
        partial(DPLRCovariance.from_data, rank=2, iterations=1),
        partial(KroneckerCovariance.from_data, rank=0),
        partial(KroneckerCovariance.from_data, rank=1),
        partial(KroneckerCovariance.from_data, rank=2, iterations=1),
    ],
)
@pytest.mark.parametrize("shape", [(5,), (3, 5)])
@pytest.mark.parametrize("batch", [(), (256,), (16, 16)])
def test_covariances(
    covariance_fn: Callable[[Tensor], Covariance],
    shape: Sequence[int],
    batch: Sequence[int],
) -> None:
    features = math.prod(shape)

    X = torch.randn(1024, features)
    A = torch.randn(features, *shape)
    X = torch.einsum("ni,i...->n...", X, A)

    cov = covariance_fn(X)

    x = torch.randn(*batch, *shape)

    # __call__
    assert x.shape == cov(x).shape

    # inv
    assert x.shape == cov.inv(x).shape

    assert torch.allclose(x, cov.inv(cov(x)))
    assert torch.allclose(x, cov(cov.inv(x)))
    assert torch.allclose(cov(x), cov.inv.inv(x))

    # color
    I = torch.eye(features)
    M = cov.color(I)
    C = cov(I)

    assert torch.allclose(C, M.T @ M)

    # logdet
    if not isinstance(cov, IsotropicCovariance):
        assert torch.allclose(cov.logdet(), -cov.inv.logdet())


@pytest.mark.parametrize("features", [8])
@pytest.mark.parametrize("rank", [1, 2])
def test_dplr_em_iterations(features: int, rank: int) -> None:
    # Generate data with non-uniform diagonal noise so that
    # PCA initialization and factor analysis give different results
    D_true = torch.rand(features) + 0.01
    V_true = torch.randn(features, rank)
    C_true = torch.diag(D_true) + V_true @ V_true.T

    L = torch.linalg.cholesky(C_true)
    X = torch.randn(1024, features) @ L.T

    X = X - X.mean(dim=0)

    def log_prob(cov: DPLRCovariance) -> torch.Tensor:
        return (
            -0.5
            * (
                features * math.log(2 * math.pi)
                + cov.logdet()
                + torch.einsum("nf,nf->n", X, cov.inv(X))
            ).mean()
        )

    log_ps = [log_prob(DPLRCovariance.from_data(X, rank=rank, iterations=i)) for i in (0, 1, 3, 7)]

    for i in range(1, len(log_ps)):
        assert log_ps[i] > log_ps[i - 1]
