r"""Utilities for modules and networks."""

import torch
import torch.nn as nn

from torch import Tensor
from typing import List, Optional, Sequence, Union


class FlattenWrapper(nn.Module):
    r"""Creates a flatten/unflatten wrapper around a backbone.

    The purpose of this module is to create a flatten/unflatten frontier between
    :mod:`azula` components that opperate on one-dimensional vectors and a backbone that
    opperates on multi-dimensional tensors, like :math:`C \times H \times W` images.

    Arguments:
        wrappee: The wrapped backbone.
        shape: The tensor shape.
    """

    def __init__(
        self,
        wrappee: nn.Module,
        shape: Optional[Sequence[int]] = None,
    ):
        super().__init__()

        self.wrappee = wrappee

        if shape is None:
            self.shape = None
        else:
            self.shape = tuple(shape)

    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        shape: Optional[Sequence[int]] = None,
        **kwargs,
    ) -> Union[Tensor, List[Tensor]]:
        r"""
        Arguments:
            x_t: A noisy vector :math:`x_t`, with shape :math:`(*, D)`.
            t: The time :math:`t`, with shape :math:`(*)`.
            shape: The unflattened tensor shape. If :py:`None`, use :py:`self.shape` instead.
            kwargs: Optional keyword arguments.

        Returns:
            The output vector(s), with shape :math:`(*, D)`.
        """

        if shape is None:
            shape = self.shape

        *batch, _ = x_t.shape

        x_t = x_t.unflatten(-1, shape)

        while t.ndim < len(batch):
            t = t.unsqueeze(0)

        y = self.wrappee(x_t, t, **kwargs)

        if torch.is_tensor(y):
            y = y.flatten(-len(shape))
        else:
            y = [z.flatten(-len(shape)) for z in y]

        return y
