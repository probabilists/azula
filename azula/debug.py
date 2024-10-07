r"""Utilities for debugging."""

__all__ = [
    "RaiseMock",
]

from unittest.mock import Mock


class RaiseMock(Mock):
    r"""Creates an object that raises an error whenever it or its attributes are called.

    Arguments:
        error: The error to be raised.
    """

    def __init__(self, error: Exception, **kwargs):
        super().__init__(side_effect=error, **kwargs)

    def _get_child_mock(self, **kwargs) -> Mock:
        return super()._get_child_mock(error=self.side_effect, **kwargs)
