r"""Covariance matrices."""

from __future__ import annotations

__all__ = [
    "Covariance",
    "IsotropicCovariance",
    "DiagonalCovariance",
    "FullCovariance",
    "DPLRCovariance",
    "DMLRCovariance",
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

    @property
    @abc.abstractmethod
    def shape(self) -> Sequence[int]:
        pass

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

    @abc.abstractmethod
    def color(self, x: Tensor) -> Tensor:
        pass

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

    lmbda: Tensor | float

    def __init__(self, lmbda: Tensor | float) -> None:
        if torch.is_tensor(lmbda):
            self.lmbda = lmbda.reshape(())
        else:
            self.lmbda = lmbda

    @property
    def shape(self) -> Sequence[int]:
        raise NotImplementedError("IsotropicCovariance's shape is ambiguous.")

    @staticmethod
    @torch.no_grad()
    def from_data(X: Tensor) -> IsotropicCovariance:
        return IsotropicCovariance(torch.var(X))

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

    def color(self, x: Tensor) -> Tensor:
        if torch.is_tensor(self.lmbda):
            return torch.sqrt(self.lmbda) * x
        else:
            return math.sqrt(self.lmbda) * x

    @property
    def inv(self) -> IsotropicCovariance:
        return IsotropicCovariance(1 / self.lmbda)

    def logdet(self) -> Tensor:
        raise NotImplementedError("IsotropicCovariance's log determinant is ambiguous.")


class DiagonalCovariance(Covariance):
    r"""Diagonal covariance matrix.

    .. math:: C = \mathrm{diag}(D)
    """

    D: Tensor

    def __init__(self, D: Tensor) -> None:
        self.D = D

    @property
    def shape(self) -> Sequence[int]:
        return self.D.shape

    @staticmethod
    @torch.no_grad()
    def from_data(X: Tensor) -> DiagonalCovariance:
        return DiagonalCovariance(torch.var(X, dim=0))

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
        y = x.reshape(-1, *self.shape)
        y = self.D * y
        return y.reshape_as(x)

    def color(self, x: Tensor) -> Tensor:
        y = x.reshape(-1, *self.shape)
        y = torch.sqrt(self.D) * y
        return y.reshape_as(x)

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
        self.Q, self.L = Q, L

    @property
    def shape(self) -> Sequence[int]:
        return self.Q.shape[:-1]

    @staticmethod
    @torch.no_grad()
    def from_data(X: Tensor) -> FullCovariance:
        samples, *shape = X.shape
        features = math.prod(shape)

        assert features < samples

        X = X.flatten(1)

        C = torch.cov(X.T).reshape(features, features)
        L, Q = torch.linalg.eigh(C)

        return FullCovariance(Q.reshape(*shape, features), L)

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
        y = x.reshape(-1, *self.shape)
        y = torch.einsum("...i,n...->ni", self.Q, y)
        y = self.L * y
        y = torch.einsum("...i,ni->n...", self.Q, y)
        return y.reshape_as(x)

    def color(self, x: Tensor) -> Tensor:
        y = x.reshape(-1, self.Q.shape[-1])
        y = torch.sqrt(self.L) * y
        y = torch.einsum("...i,ni->n...", self.Q, y)
        return y.reshape_as(x)

    @property
    def inv(self) -> FullCovariance:
        return FullCovariance(self.Q, 1 / self.L)

    def logdet(self) -> Tensor:
        return torch.log(self.L).sum()


class DPLRCovariance(Covariance):
    r"""Diagonal plus low-rank (DPLR) covariance matrix.

    .. math:: \mathrm{diag}(D) + V V^\top

    Wikipedia:
        https://wikipedia.org/wiki/Low-rank_approximation
    """

    D: Tensor
    V: Tensor

    def __init__(self, D: Tensor, V: Tensor) -> None:
        self.D, self.V = D, V

    @property
    def shape(self) -> Sequence[int]:
        return self.D.shape

    @property
    def rank(self) -> int:
        return self.V.shape[-1]

    @staticmethod
    @torch.no_grad()
    def from_data(X: Tensor, rank: int = 1, iterations: int = 0) -> DPLRCovariance:
        """
        References:
            | The EM Algorithm for Mixtures of Factor Analyzers (Zoubin et al., 1996)
            | https://mlg.eng.cam.ac.uk/zoubin/papers/tr-96-1.pdf
        """

        samples, *shape = X.shape
        features = math.prod(shape)

        assert 0 < rank < min(features, samples)

        X = X.flatten(1)
        X = X - X.mean(dim=0)

        # PCA initialization
        if samples < features:
            C = torch.einsum("if,jf->ij", X, X) / (samples - 1)
        else:
            C = torch.einsum("ni,nj->ij", X, X) / (samples - 1)

        if 3 * rank < min(samples, features):
            L, Q = torch.lobpcg(C, k=rank)
        else:
            L, Q = torch.linalg.eigh(C)
            L, Q = L[-rank:], Q[:, -rank:]

        if samples < features:
            Q = torch.einsum("ni,nj->ij", X, Q)
            Q = Q / torch.linalg.norm(Q, dim=0, keepdim=True)

        V = Q * torch.sqrt(L)
        D = torch.var(X, dim=0) - torch.einsum("fi,fi->f", V, V)

        # EM iterations for factor analysis
        for _ in range(iterations):
            B = DPLRCovariance(D, V).inv(V.T)
            Ez = torch.einsum("if,nf->ni", B, X)
            Ezz = (
                torch.eye(V.shape[-1], dtype=D.dtype, device=D.device)
                - torch.einsum("if,fj->ij", B, V)
                + torch.einsum("ni,nj->ij", Ez, Ez) / (samples - 1)
            )
            Ezz_inv = torch.cholesky_inverse(torch.linalg.cholesky(Ezz))

            V = torch.einsum("nf,ni,ij->fj", X, Ez, Ezz_inv) / (samples - 1)
            D = torch.var(X, dim=0) - torch.einsum("fi,ni,nf->f", V, Ez, X) / (samples - 1)

        return DPLRCovariance(D.reshape(shape), V.reshape(*shape, -1))

    def __add__(self, other: Covariance) -> DPLRCovariance:
        if isinstance(other, IsotropicCovariance):
            return DPLRCovariance(self.D + other.lmbda, self.V)
        elif isinstance(other, DiagonalCovariance):
            return DPLRCovariance(self.D + other.D, self.V)
        elif isinstance(other, DPLRCovariance):
            return DPLRCovariance(
                self.D + other.D,
                torch.cat((self.V, other.V), dim=-1),
            )
        else:
            return NotImplemented

    def __mul__(self, other: Covariance) -> DPLRCovariance:
        if isinstance(other, IsotropicCovariance):
            return DPLRCovariance(
                self.D * other.lmbda,
                self.V * torch.sqrt(other.lmbda),
            )
        else:
            return NotImplemented

    def __matmul__(self, x: Tensor) -> Tensor:
        y = x.reshape(-1, *self.shape)
        y = self.D * y + torch.einsum(
            "...i,ni->n...", self.V, torch.einsum("...i,n...->ni", self.V, y)
        )
        return y.reshape_as(x)

    def color(self, x: Tensor) -> Tensor:
        W = torch.einsum("...,...i->...i", torch.rsqrt(self.D), self.V)
        L, Q = torch.linalg.eigh(torch.einsum("...i,...j->ij", W, W))
        U = torch.einsum("...i,ij,j->...j", W, Q, torch.rsqrt(L))

        y = x.reshape(-1, *self.shape)
        y = y + torch.einsum(
            "...i,i,ni->n...",
            U,
            torch.sqrt(1 + L) - 1,
            torch.einsum("...i,n...->ni", U, y),
        )
        y = torch.sqrt(self.D) * y

        return y.reshape_as(x)

    @property
    def K(self) -> Tensor:  # capacitance
        return torch.eye(self.rank, dtype=self.D.dtype, device=self.D.device) + torch.einsum(
            "...i,...,...j->ij",
            self.V,
            1 / self.D,
            self.V,
        )

    @property
    def inv(self) -> DMLRCovariance:
        D = 1 / self.D
        L, Q = torch.linalg.eigh(self.K)
        V = torch.einsum("...,...i,ij,j->...j", D, self.V, Q, torch.rsqrt(L))

        return DMLRCovariance(D, V)

    def logdet(self) -> Tensor:
        return torch.log(self.D).sum() + torch.linalg.slogdet(self.K).logabsdet


class DMLRCovariance(Covariance):
    r"""Diagonal minus low-rank (DMLR) covariance matrix.

    .. math:: \mathrm{diag}(D) - V V^\top
    """

    D: Tensor
    V: Tensor

    def __init__(self, D: Tensor, V: Tensor) -> None:
        self.D, self.V = D, V

    @property
    def shape(self) -> Sequence[int]:
        return self.D.shape

    @property
    def rank(self) -> int:
        return self.V.shape[-1]

    def __add__(self, other: Covariance) -> DMLRCovariance:
        if isinstance(other, IsotropicCovariance):
            return DMLRCovariance(self.D + other.lmbda, self.V)
        elif isinstance(other, DiagonalCovariance):
            return DMLRCovariance(self.D + other.D, self.V)
        elif isinstance(other, DMLRCovariance):
            return DMLRCovariance(
                self.D + other.D,
                torch.cat((self.V, other.V), dim=-1),
            )
        else:
            return NotImplemented

    def __mul__(self, other: Covariance) -> DMLRCovariance:
        if isinstance(other, IsotropicCovariance):
            return DMLRCovariance(
                self.D * other.lmbda,
                self.V * torch.sqrt(other.lmbda),
            )
        else:
            return NotImplemented

    def __matmul__(self, x: Tensor) -> Tensor:
        y = x.reshape(-1, *self.shape)
        y = self.D * y - torch.einsum(
            "...i,ni->n...", self.V, torch.einsum("...i,n...->ni", self.V, y)
        )
        return y.reshape_as(x)

    def color(self, x: Tensor) -> Tensor:
        W = torch.einsum("...,...i->...i", torch.rsqrt(self.D), self.V)
        L, Q = torch.linalg.eigh(torch.einsum("...i,...j->ij", W, W))
        U = torch.einsum("...i,ij,j->...j", W, Q, torch.rsqrt(L))

        y = x.reshape(-1, *self.shape)
        y = y + torch.einsum(
            "...i,i,ni->n...",
            U,
            torch.sqrt(1 - L) - 1,
            torch.einsum("...i,n...->ni", U, y),
        )
        y = torch.sqrt(self.D) * y

        return y.reshape_as(x)

    @property
    def K(self) -> Tensor:  # capacitance
        return torch.eye(self.rank, dtype=self.D.dtype, device=self.D.device) - torch.einsum(
            "...i,...,...j->ij",
            self.V,
            1 / self.D,
            self.V,
        )

    @property
    def inv(self) -> DPLRCovariance:
        D = 1 / self.D
        L, Q = torch.linalg.eigh(self.K)
        V = torch.einsum("...,...i,ij,j->...j", D, self.V, Q, torch.rsqrt(L))

        return DPLRCovariance(D, V)

    def logdet(self) -> Tensor:
        return torch.log(self.D).sum() + torch.linalg.slogdet(self.K).logabsdet


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

    @property
    def shape(self) -> Sequence[int]:
        return tuple(Q.shape[0] for Q in self.Qs)

    @staticmethod
    @torch.no_grad()
    def from_data(X: Tensor, rank: int = 0, iterations: int = 0) -> KroneckerCovariance:
        Qs = []

        for i in range(1, X.ndim):
            Ci = torch.cov(X.movedim(i, 0).flatten(1))
            _, Qi = torch.linalg.eigh(Ci)
            Qs.append(Qi)

        abc = string.ascii_lowercase[: len(Qs)]

        X = torch.einsum(f"...{abc}," + ",".join(f"{i}{i.upper()}" for i in abc), X, *Qs)

        if rank > 0 and len(Qs) > 1:
            L = DPLRCovariance.from_data(X, rank=rank, iterations=iterations)
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
        y = x.reshape(-1, *self.shape)

        abc = string.ascii_lowercase[: len(self.Qs)]

        y = torch.einsum(f"...{abc}," + ",".join(f"{i}{i.upper()}" for i in abc), y, *self.Qs)
        y = self.L @ y
        y = torch.einsum(f"...{abc}," + ",".join(f"{i.upper()}{i}" for i in abc), y, *self.Qs)

        return y.reshape_as(x)

    def color(self, x: Tensor) -> Tensor:
        y = x.reshape(-1, *self.shape)

        abc = string.ascii_lowercase[: len(self.Qs)]

        y = self.L.color(y)
        y = torch.einsum(f"...{abc}," + ",".join(f"{i.upper()}{i}" for i in abc), y, *self.Qs)

        return y.reshape_as(x)

    @property
    def inv(self) -> KroneckerCovariance:
        return KroneckerCovariance(self.Qs, self.L.inv)

    def logdet(self) -> Tensor:
        return self.L.logdet()
