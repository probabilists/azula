r"""Covariance matrices."""

from __future__ import annotations

__all__ = [
    "Covariance",
    "IsotropicCovariance",
    "DiagonalCovariance",
    "FullCovariance",
    "DPLRCovariance",
    "KroneckerCovariance",
]

import abc
import math
import string
import torch

from collections.abc import Sequence
from torch import Tensor


class Covariance(abc.ABC):
    r"""Abstract covariance matrix."""

    @abc.abstractmethod
    def __add__(self, other: Covariance) -> Covariance:
        pass

    def __radd__(self, other: Covariance) -> Covariance:
        return self.__add__(other)

    @abc.abstractmethod
    def __mul__(self, other: Covariance) -> Covariance:
        pass

    def __rmul__(self, other: Covariance) -> Covariance:
        return self.__mul__(other)

    @abc.abstractmethod
    def __matmul__(self, x: Tensor) -> Tensor:
        pass

    def __call__(self, x: Tensor) -> Tensor:
        return self.__matmul__(x)

    @property
    @abc.abstractmethod
    def inv(self) -> Covariance:
        pass

    @abc.abstractmethod
    def logdet(self) -> Tensor:
        pass

    def to(self, *args, **kwargs) -> Covariance:
        new = object.__new__(type(self))

        for k, v in self.__dict__.items():
            if hasattr(v, "to"):
                new.__dict__[k] = v.to(*args, **kwargs)
            elif isinstance(v, list | tuple):
                new.__dict__[k] = type(v)(
                    w.to(*args, **kwargs) if hasattr(w, "to") else w for w in v
                )
            else:
                new.__dict__[k] = v

        return new

    def is_floating_point(self) -> bool:  # used by `nn.Module.to(dtype)`
        return True


class IsotropicCovariance(Covariance):
    r"""Isotropic covariance matrix.

    .. math:: C = \lambda I
    """

    lmbda: Tensor

    def __init__(self, lmbda: Tensor) -> None:
        self.lmbda = lmbda.reshape(())

    @classmethod
    @torch.no_grad()
    def from_data(self, X: Tensor) -> IsotropicCovariance:
        return IsotropicCovariance(X.var())

    def __add__(self, other: Covariance) -> IsotropicCovariance:
        if isinstance(other, IsotropicCovariance):
            return IsotropicCovariance(self.lmbda + other.lmbda)
        else:
            return NotImplemented

    def __mul__(self, other: Covariance) -> IsotropicCovariance:
        if isinstance(other, IsotropicCovariance):
            return IsotropicCovariance(self.lmbda * other.lmbda)
        else:
            return NotImplemented

    def __matmul__(self, x: Tensor) -> Tensor:
        return self.lmbda * x

    @property
    def inv(self) -> IsotropicCovariance:
        return IsotropicCovariance(1 / self.lmbda)

    def logdet(self) -> Tensor:
        return torch.log(self.lmbda)


class DiagonalCovariance(Covariance):
    r"""Diagonal covariance matrix.

    .. math:: C = \mathrm{diag}(D)
    """

    D: Tensor

    def __init__(self, D: Tensor) -> None:
        self.D = D

    @classmethod
    @torch.no_grad()
    def from_data(self, X: Tensor) -> DiagonalCovariance:
        return DiagonalCovariance(X.var(dim=0))

    def __add__(self, other: Covariance) -> DiagonalCovariance:
        if isinstance(other, IsotropicCovariance):
            return DiagonalCovariance(self.D + other.lmbda)
        if isinstance(other, DiagonalCovariance):
            return DiagonalCovariance(self.D + other.D)
        else:
            return NotImplemented

    def __mul__(self, other: Covariance) -> DiagonalCovariance:
        if isinstance(other, IsotropicCovariance):
            return DiagonalCovariance(self.D * other.lmbda)
        if isinstance(other, DiagonalCovariance):
            return DiagonalCovariance(self.D * other.D)
        else:
            return NotImplemented

    def __matmul__(self, x: Tensor) -> Tensor:
        return self.D * x

    @property
    def inv(self) -> DiagonalCovariance:
        return DiagonalCovariance(1 / self.D)

    def logdet(self) -> Tensor:
        return torch.log(self.D).sum()


class FullCovariance(Covariance):
    r"""Full covariance matrix.

    .. math:: C = Q \mathrm{diag}(L) Q^\top

    where :math:`Q` is an orthonormal matrix.
    """

    Q: Tensor
    L: Tensor

    def __init__(self, Q: Tensor, L: Tensor) -> None:
        self.Q = Q
        self.L = L

    @classmethod
    @torch.no_grad()
    def from_data(self, X: Tensor) -> FullCovariance:
        samples, *shape = X.shape
        features = math.prod(shape)

        assert features < samples

        X = X.flatten(1)

        C = torch.cov(X.T)
        L, Q = torch.linalg.eigh(C)

        return FullCovariance(Q.reshape(*shape, *shape), L.reshape(shape))

    def __add__(self, other: Covariance) -> FullCovariance:
        if isinstance(other, IsotropicCovariance):
            return FullCovariance(self.Q, self.L + other.lmbda)
        else:
            return NotImplemented

    def __mul__(self, other: Covariance) -> FullCovariance:
        if isinstance(other, IsotropicCovariance):
            return FullCovariance(self.Q, self.L * other.lmbda)
        else:
            return NotImplemented

    def __matmul__(self, x: Tensor) -> Tensor:
        X = x.reshape(-1, *self.L.shape)

        abc = string.ascii_lowercase[: self.L.ndim]

        X = torch.einsum(f"...{abc},{abc}{abc.upper()}", X, self.Q)
        X = self.L * X
        X = torch.einsum(f"...{abc},{abc.upper()}{abc}", X, self.Q)

        return X.reshape_as(x)

    @property
    def inv(self) -> FullCovariance:
        return FullCovariance(self.Q, 1 / self.L)

    def logdet(self) -> Tensor:
        return torch.log(self.L).sum()


class DPLRCovariance(Covariance):
    r"""Diagonal plus low-rank (DPLR) covariance matrix.

    .. math:: \mathrm{diag}(D) + V \mathrm{diag}(S) V^\top

    Wikipedia:
        https://wikipedia.org/wiki/Low-rank_approximation
    """

    D: Tensor
    V: Tensor
    S: Tensor

    def __init__(self, D: Tensor, V: Tensor, S: Tensor | None = None) -> None:
        self.D, self.V = D, V

        if S is None:
            self.S = V.new_ones(self.rank)
        else:
            self.S = S

    @classmethod
    @torch.no_grad()
    def from_data(self, X: Tensor, rank: int = 1) -> DPLRCovariance:
        """
        References:
            | Mixtures of probabilistic principal component analysers (Tipping and Bishop, 1999)
            | https://www.miketipping.com/abstracts.htm#Tipping:NC98
        """

        samples, *shape = X.shape
        features = math.prod(shape)

        assert rank < min(features, samples)

        X = X.flatten(1)
        X = X - X.mean(dim=0)

        if samples < features:
            C = torch.einsum("ik,jk->ij", X, X) / samples
        else:
            C = torch.einsum("ki,kj->ij", X, X) / samples

        if 3 * rank < min(samples, features):
            L, Q = torch.lobpcg(C, k=rank)
        else:
            L, Q = torch.linalg.eigh(C)
            L, Q = L[-rank:], Q[:, -rank:]

        if samples < features:
            V = torch.einsum("ij,ik->kj", X, Q)
            V = V / torch.linalg.norm(V, dim=1, keepdim=True)
        else:
            V = Q.T

        D = torch.clip(torch.trace(C) - torch.sum(L), min=0.0) / (features - rank)
        S = torch.clip(L - D, min=0.0)

        return DPLRCovariance(D.expand(shape), V.reshape(-1, *shape), S)

    def __add__(self, other: Covariance) -> DPLRCovariance:
        if isinstance(other, IsotropicCovariance):
            return DPLRCovariance(self.D + other.lmbda, self.V, self.S)
        elif isinstance(other, DiagonalCovariance):
            return DPLRCovariance(self.D + other.D, self.V, self.S)
        elif isinstance(other, DPLRCovariance):
            return DPLRCovariance(
                self.D + other.D,
                torch.cat((self.V, other.V)),
                torch.cat((self.S, other.S)),
            )
        else:
            return NotImplemented

    def __mul__(self, other: Covariance) -> DiagonalCovariance:
        if isinstance(other, IsotropicCovariance):
            return DPLRCovariance(self.D * other.lmbda, self.V, self.S * other.lmbda)
        else:
            return NotImplemented

    def __matmul__(self, x: Tensor) -> Tensor:
        X = x.reshape(-1, *self.D.shape)
        X = self.D * X + torch.einsum(
            "i...,i,ni->n...", self.V, self.S, torch.einsum("i...,n...->ni", self.V, X)
        )
        return X.reshape_as(x)

    @property
    def rank(self) -> int:
        return self.V.shape[0]

    @property
    def K(self) -> Tensor:  # capacitance
        return torch.diag(1 / self.S) + torch.einsum(
            "i...,...,j...->ij", self.V, 1 / self.D, self.V
        )

    @property
    def inv(self) -> DPLRCovariance:
        D = 1 / self.D
        L, Q = torch.linalg.eigh(self.K)
        V = torch.einsum("...,i...,ij->j...", D, self.V, Q)
        S = -1 / L

        return DPLRCovariance(D, V, S)

    def logdet(self) -> Tensor:  # TODO: add tests
        return (
            torch.log(self.D).sum()
            + torch.log(torch.abs(self.S)).sum()
            + torch.linalg.slogdet(self.K).logabsdet
        )


class KroneckerCovariance(Covariance):
    r"""Kronecker-factorized covariance matrix.

    .. math:: C = (Q_1 \otimes \dots \otimes Q_n) \, L \, (Q_1 \otimes \dots \otimes Q_n)^\top

    where :math:`Q_i` are orthonormal matrices for each dimension and :math:`\otimes` denotes the Kronecker product.

    Wikipedia:
        https://wikipedia.org/wiki/Kronecker_product
    """

    Qs: Sequence[Tensor]
    L: Covariance

    def __init__(self, Qs: Sequence[Tensor], L: Covariance) -> None:
        self.Qs = tuple(Qs)
        self.L = L

    @classmethod
    @torch.no_grad()
    def from_data(self, X: Tensor, rank: int = 0) -> KroneckerCovariance:
        Qs = []

        for i in range(1, X.ndim):
            Ci = torch.cov(X.movedim(i, 0).flatten(1))
            _, Qi = torch.linalg.eigh(Ci)
            Qs.append(Qi)

        abc = string.ascii_lowercase[: len(Qs)]

        X = torch.einsum(f"...{abc}," + ",".join(f"{i}{i.upper()}" for i in abc), X, *Qs)

        if rank > 0:
            L = DPLRCovariance.from_data(X, rank=rank)
        else:
            L = DiagonalCovariance.from_data(X)

        return KroneckerCovariance(Qs, L)

    def __add__(self, other: Covariance) -> KroneckerCovariance:
        if isinstance(other, IsotropicCovariance):
            return KroneckerCovariance(self.Qs, self.L + other)
        else:
            return NotImplemented

    def __mul__(self, other: Covariance) -> KroneckerCovariance:
        if isinstance(other, IsotropicCovariance):
            return KroneckerCovariance(self.Qs, self.L * other)
        else:
            return NotImplemented

    def __matmul__(self, x: Tensor) -> Tensor:
        X = x.reshape(-1, *(Q.shape[0] for Q in self.Qs))

        abc = string.ascii_lowercase[: len(self.Qs)]

        X = torch.einsum(f"...{abc}," + ",".join(f"{i}{i.upper()}" for i in abc), X, *self.Qs)
        X = self.L @ X
        X = torch.einsum(f"...{abc}," + ",".join(f"{i.upper()}{i}" for i in abc), X, *self.Qs)

        return X.reshape_as(x)

    @property
    def inv(self) -> KroneckerCovariance:
        return KroneckerCovariance(self.Qs, self.L.inv)

    def logdet(self) -> Tensor:
        return self.L.logdet()
