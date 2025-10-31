r"""Tests configuration."""

import pytest
import torch


def pytest_addoption(parser):
    parser.addoption("--device", type=str, default="cpu")
    parser.addoption("--dtype", type=str, default="float32")


@pytest.fixture(autouse=True, scope="module")
def torch_device(pytestconfig):
    device = pytestconfig.getoption("device")
    device = torch.device(device)

    default = torch.get_default_device()

    if device is default:
        yield
    else:
        try:
            yield torch.set_default_device(device)
        finally:
            torch.set_default_device(default)


@pytest.fixture(autouse=True, scope="module")
def torch_dtype(pytestconfig):
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
