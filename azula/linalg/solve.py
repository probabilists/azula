r"""Linear system solvers."""

__all__ = [
    "cg",
    "gmres",
]

import torch

from torch import Tensor
from typing import Callable, Optional


def cg(
    A: Callable[[Tensor], Tensor],
    b: Tensor,
    x0: Optional[Tensor] = None,
    iterations: int = 1,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    r"""Solves a linear system :math:`Ax = b` with conjugate gradient (CG) iterations.

    The matrix :math:`A \in \mathbb{R}^{D \times D}` must be symmetric positive
    (semi)definite.

    Wikipedia:
        https://wikipedia.org/wiki/Conjugate_gradient_method

    Warning:
        This function is optimized for GPU execution. To avoid CPU-GPU communication,
        all iterations are performed regardless of convergence.

    Arguments:
        A: The linear operator :math:`x \mapsto Ax`.
        b: The right-hand side vector :math:`b`, with shape :math:`(*, D)`.
        x0: An initial guess :math:`x_0`, with shape :math:`(*, D)`. If :py:`None`, use
            :math:`x_0 = 0` instead.
        iterations: The number of CG iterations :math:`n`.
        dtype: The data type used for intermediate computations. If :py:`None`, use
            :class:`torch.float64` instead.

    Returns:
        The :math:`n`-th iteration :math:`x_n`, with shape :math:`(*, D)`.
    """

    if dtype is None:
        dtype = torch.float64

    epsilon = torch.finfo(dtype).smallest_normal

    if x0 is None:
        x = torch.zeros_like(b)
        r = b
    else:
        x = x0
        r = b - A(x0)

    x = x.to(dtype)
    r = r.to(dtype)
    rr = torch.einsum("...i,...i", r, r)
    p = r

    for _ in range(iterations):
        Ap = A(p.to(b)).to(dtype)
        pAp = torch.einsum("...i,...i", p, Ap)
        alpha = rr / torch.clip(pAp, min=epsilon)
        x_ = x + alpha[..., None] * p
        r_ = r - alpha[..., None] * Ap
        rr_ = torch.einsum("...i,...i", r_, r_)
        beta = rr_ / torch.clip(rr, min=epsilon)
        p_ = r_ + beta[..., None] * p

        x, r, rr, p = x_, r_, rr_, p_

    return x.to(b)


def gmres(
    A: Callable[[Tensor], Tensor],
    b: Tensor,
    x0: Optional[Tensor] = None,
    iterations: int = 1,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    r"""Solves a linear system :math:`Ax = b` with generalized minimal residual (GMRES) iterations.

    The matrix :math:`A \in \mathbb{R}^{D \times D}` can be non-symmetric non-definite.

    Wikipedia:
        https://wikipedia.org/wiki/Generalized_minimal_residual_method

    Warning:
        This function is optimized for GPU execution. To avoid CPU-GPU communication,
        all iterations are performed regardless of convergence.

    Arguments:
        A: The linear operator :math:`x \mapsto Ax`.
        b: The right-hand side vector :math:`b`, with shape :math:`(*, D)`.
        x0: An initial guess :math:`x_0`, with shape :math:`(*, D)`. If :py:`None`, use
            :math:`x_0 = 0` instead.
        iterations: The number of GMRES iterations :math:`n`.
        dtype: The data type used for intermediate computations. If :py:`None`, use
            :class:`torch.float64` instead.

    Returns:
        The :math:`n`-th iteration :math:`x_n`, with shape :math:`(*, D)`.
    """

    if dtype is None:
        dtype = torch.float64

    epsilon = torch.finfo(dtype).smallest_normal

    if x0 is None:
        r = b
    else:
        r = b - A(x0)

    r = r.to(dtype)

    def normalize(x):
        norm = torch.linalg.vector_norm(x, dim=-1)
        x = x / torch.clip(norm[..., None], min=epsilon)

        return x, norm

    def rotation(a, b):
        c = torch.clip(torch.sqrt(a * a + b * b), min=epsilon)
        return a / c, -b / c

    V = [None for _ in range(iterations + 1)]
    B = [None for _ in range(iterations + 1)]
    H = [[None for _ in range(iterations)] for _ in range(iterations + 1)]
    cs = [None for _ in range(iterations)]
    ss = [None for _ in range(iterations)]

    V[0], B[0] = normalize(r)

    for j in range(iterations):
        v = V[j].to(b)
        w = A(v).to(dtype)

        # Apply Arnoldi iteration to get the j+1-th basis
        for i in range(j + 1):
            H[i][j] = torch.einsum("...i,...i", w, V[i])
            w = w - H[i][j][..., None] * V[i]
        w, w_norm = normalize(w)
        H[j + 1][j] = w_norm
        V[j + 1] = w

        # Apply Givens rotation
        for i in range(j):
            tmp = cs[i] * H[i][j] - ss[i] * H[i + 1][j]
            H[i + 1][j] = cs[i] * H[i + 1][j] + ss[i] * H[i][j]
            H[i][j] = tmp

        cs[j], ss[j] = rotation(H[j][j], H[j + 1][j])
        H[j][j] = cs[j] * H[j][j] - ss[j] * H[j + 1][j]

        # Update residual vector
        B[j + 1] = ss[j] * B[j]
        B[j] = cs[j] * B[j]

        # Fill with zeros
        for i in range(j + 1, iterations + 1):
            H[i][j] = torch.zeros_like(H[j][j])

    V, B, H = V[:-1], B[:-1], H[:-1]

    V = torch.stack(V, dim=-2)
    B = torch.stack(B, dim=-1)
    H = torch.stack([torch.stack(Hi, dim=-1) for Hi in H], dim=-2)

    y = torch.linalg.solve_triangular(
        H + epsilon * torch.eye(iterations, dtype=dtype, device=H.device),
        B.unsqueeze(dim=-1),
        upper=True,
    ).squeeze(dim=-1)

    if x0 is None:
        x = torch.einsum("...ij,...i", V, y)
    else:
        x = x0 + torch.einsum("...ij,...i", V, y)

    return x.to(b)
