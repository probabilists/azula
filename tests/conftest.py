r"""Tests configuration."""

import argparse
import pytest
import torch

from collections.abc import Iterator


def pytest_addoption(parser: argparse.ArgumentParser) -> None:
    parser.addoption("--device", type=str, default="cpu")
    parser.addoption("--dtype", type=str, default="float32")


@pytest.fixture(autouse=True, scope="module")
def torch_device(pytestconfig: pytest.Config) -> Iterator[None]:
    device = pytestconfig.getoption("device")
    device = torch.device(device)

    if hasattr(torch, "get_default_device"):
        default = torch.get_default_device()
    else:
        default = torch.empty(()).device

    if device == default:
        yield
    else:
        try:
            yield torch.set_default_device(device)
        finally:
            torch.set_default_device(default)


@pytest.fixture(autouse=True, scope="module")
def torch_dtype(pytestconfig: pytest.Config) -> Iterator[None]:
    dtype = pytestconfig.getoption("dtype")

    if dtype == "float32":
        dtype = torch.float32
    elif dtype == "float64":
        dtype = torch.float64
    else:
        raise NotImplementedError()

    default = torch.get_default_dtype()

    try:
        yield torch.set_default_dtype(dtype)
    finally:
        torch.set_default_dtype(default)
