r"""Covariance matrices."""

from __future__ import annotations

__all__ = [
    "Covariance",
    "DiagonalCovariance",
    "IsotropicCovariance",
    "DPLRCovariance",
    "PreconditionedCovariance",
]

import abc
import math
import torch

from torch import Tensor
from typing import Sequence


class Covariance(abc.ABC):
    r"""Abstract covariance matrix."""

    @abc.abstractmethod
    def __add__(self, other: Covariance) -> Covariance:
        pass

    @abc.abstractmethod
    def __mul__(self, other: Covariance) -> Covariance:
        pass

    @abc.abstractmethod
    def __matmul__(self, x: Tensor) -> Tensor:
        pass

    def __call__(self, x: Tensor) -> Tensor:
        return self.__matmul__(x)

    @property
    @abc.abstractmethod
    def inv(self) -> Covariance:
        pass

    def to(self, **kwargs) -> Covariance:
        new = object.__new__(type(self))

        for k, v in self.__dict__.items():
            if hasattr(v, "to"):
                new.__dict__[k] = v.to(**kwargs)
            elif isinstance(v, (list, tuple)):
                new.__dict__[k] = type(v)(w.to(**kwargs) if hasattr(w, "to") else w for w in v)
            else:
                new.__dict__[k] = v

        return new


class IsotropicCovariance(Covariance):
    r"""Isotropic covariance matrix."""

    lmbda: Tensor

    def __init__(self, lmbda: Tensor):
        self.lmbda = lmbda.reshape(())

    @classmethod
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


class DiagonalCovariance(Covariance):
    r"""Diagonal covariance matrix."""

    D: Tensor

    def __init__(self, D: Tensor):
        self.D = D

    @classmethod
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


class DPLRCovariance(Covariance):
    r"""Diagonal plus low-rank (DPLR) covariance matrix."""

    D: Tensor
    V: Tensor
    S: Tensor

    def __init__(self, D: Tensor, V: Tensor, S: Tensor = None):
        self.D, self.V = D, V

        if S is None:
            self.S = V.new_ones(self.rank)
        else:
            self.S = S

    @classmethod
    def from_data(self, X: Tensor, rank: int = 1) -> DPLRCovariance:
        samples, *shape = X.shape
        features = math.prod(shape)

        X = X.flatten(1)
        X = X - X.mean(dim=0)

        if samples < features:
            C = torch.einsum("ik,jk->ij", X, X) / samples
        else:
            C = torch.einsum("ki,kj->ij", X, X) / samples

        L, Q = torch.lobpcg(C, k=rank)

        if rank < features:
            D = (torch.trace(C) - torch.sum(L)) / (features - rank)
        else:
            D = torch.tensor(1e-6).to(C)

        if samples < features:
            V = torch.einsum("ij,ik->kj", X, Q)
            V = V / torch.linalg.norm(V, dim=1, keepdim=True)
        else:
            V = Q.T

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
        X = x.reshape(-1, *self.V.shape[1:])

        X = self.D * X + torch.einsum(
            "i...,i,ni->n...", self.V, self.S, torch.einsum("i...,n...->ni", self.V, X)
        )

        return X.reshape_as(x)

    @property
    def rank(self) -> int:
        return self.V.shape[0]

    @property
    def C(self) -> Tensor:  # capacitance
        return torch.diag(1 / self.S) + torch.einsum(
            "i...,...,j...->ij", self.V, 1 / self.D, self.V
        )

    @property
    def inv(self) -> DPLRCovariance:
        D = 1 / self.D
        L, Q = torch.linalg.eigh(self.C)
        V = torch.einsum("...,i...,ij->j...", D, self.V, Q)
        S = -1 / L

        return DPLRCovariance(D, V, S)


class PreconditionedCovariance(Covariance):
    r"""Preconditioned covariance matrix."""

    C: Covariance
    Qs: Sequence[Tensor]

    def __init__(self, C: Covariance, Qs: Sequence[Tensor]):
        self.C = C
        self.Qs = tuple(Qs)

    @classmethod
    def from_data(self, X: Tensor, rank: int = 0) -> PreconditionedCovariance:
        Qs = []

        for i in range(1, X.ndim):
            Ci = torch.cov(X.movedim(i, 0).flatten(1))
            _, Qi = torch.linalg.eigh(Ci)
            Qs.append(Qi)

        for Q in Qs:
            X = torch.tensordot(X, Q, dims=[[1], [0]])

        if rank > 0:
            C = DPLRCovariance.from_data(X, rank=rank)
        else:
            C = DiagonalCovariance.from_data(X)

        return PreconditionedCovariance(C, Qs)

    def __add__(self, other: Covariance) -> PreconditionedCovariance:
        if isinstance(other, IsotropicCovariance):
            return PreconditionedCovariance(self.C + other, self.Qs)
        elif isinstance(other, PreconditionedCovariance):
            assert all(Q1 is Q2 for Q1, Q2 in zip(self.Qs, other.Qs))
            return PreconditionedCovariance(
                self.C + other.C,
                self.Qs,
            )
        else:
            return NotImplemented

    def __mul__(self, other: Covariance) -> PreconditionedCovariance:
        if isinstance(other, IsotropicCovariance):
            return PreconditionedCovariance(self.C * other, self.Qs)
        else:
            return NotImplemented

    def __matmul__(self, x: Tensor) -> Tensor:
        X = x.reshape(-1, *(Q.shape[0] for Q in self.Qs))

        for Q in self.Qs:
            X = torch.tensordot(X, Q, dims=[[1], [0]])

        X = self.C @ X

        for Q in self.Qs:
            X = torch.tensordot(X, Q, dims=[[1], [1]])

        return X.reshape_as(x)

    @property
    def inv(self) -> PreconditionedCovariance:
        return PreconditionedCovariance(self.C.inv, self.Qs)
