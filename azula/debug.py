r"""Utilities for debugging."""

from typing import Any
from unittest.mock import Mock


class RaiseMock(Mock):
    r"""Creates an object that raises an error whenever it or its children are called.

    Arguments:
        error: The error to be raised.
    """

    def __init__(self, error: Exception, **kwargs):
        super().__init__(side_effect=error, **kwargs)

    def _get_child_mock(self, **kwargs: Any) -> Mock:
        return super()._get_child_mock(error=self.side_effect, **kwargs)
