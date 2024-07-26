r"""Linear algebra."""

__all__ = [
    'conjugate_gradient',
]

import torch

from torch import Tensor
from typing import Callable


@torch.no_grad()
def conjugate_gradient(
    A: Callable[[Tensor], Tensor],
    b: Tensor,
    x: Tensor = None,
    iterations: int = None,
    dtype: torch.dtype = None,
) -> Tensor:
    r"""Solves a linear system :math:`Ax = b` with conjugate gradient iterations.

    Wikipedia:
        https://wikipedia.org/wiki/Conjugate_gradient_method

    Warning:
        This function is optimized for GPU execution. To avoid CPU-GPU communication,
        all iterations are performed regardless of convergence.

    Arguments:
        A: The linear operator :math:`x \mapsto Ax`. The matrix :math:`A \in \mathbb{R}^{D \times D}`
            must be symmetric positive-(semi)definite.
        b: The right-hand side vector :math:`b`, with shape :math:`(*, D)`.
        x: An initial guess :math:`x_0`, with shape :math:`(*, D)`. If :py:`None`, use
            :math:`x_0 = 0` instead.
        iterations: The number of CG iterations :math:`n`. If :py:`None`, use
            :math:`n = D` instead.
        dtype: The data type used for intermediate computations. If :py:`None`, use
            :class:`torch.float64` instead.

    Returns:
        The :math:`n`-th iteration :math:`x_n`, with shape :math:`(*, D)`.
    """

    *_, D = b.shape

    if iterations is None:
        iterations = D

    if dtype is None:
        dtype = torch.float64

    if x is None:
        x = torch.zeros_like(b)
        r = b
    else:
        r = b - A(x)

    x = x.to(dtype)
    r = r.to(dtype)
    rr = torch.einsum('...i,...i', r, r)
    p = r

    for _ in range(iterations):
        Ap = A(p.to(b.dtype)).to(dtype)
        pAp = torch.einsum('...i,...i', p, Ap)
        alpha = rr / pAp
        x_ = x + alpha[..., None] * p
        r_ = r - alpha[..., None] * Ap
        rr_ = torch.einsum('...i,...i', r_, r_)
        beta = rr_ / rr
        p_ = r + beta[..., None] * p

        x, r, rr, p = x_, r_, rr_, p_

    return x.to(b.dtype)
