r"""Tests for the azula.nn.unet module."""

import pytest
import torch

from pathlib import Path
from torch.torch_version import TorchVersion

from azula.nn.unet import UNet


@pytest.mark.parametrize("length", [15, 16])
@pytest.mark.parametrize("in_channels, out_channels", [(3, 5)])
@pytest.mark.parametrize("mod_features", [0, 16])
@pytest.mark.parametrize("dropout", [None, 0.1])
@pytest.mark.parametrize("spatial", [1, 2])
@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.flaky(reruns=2)
def test_UNet(
    tmp_path: Path,
    length: int,
    in_channels: int,
    out_channels: int,
    mod_features: int,
    dropout: float,
    spatial: int,
    checkpointing: bool,
    batch_size: int,
) -> None:
    make = lambda: UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        mod_features=mod_features,
        hid_channels=(16, 24, 32),
        hid_blocks=(1, 2, 3),
        dropout=dropout,
        spatial=spatial,
        checkpointing=checkpointing,
    )

    model = make()
    model.train()

    # Call
    x = torch.randn((batch_size, in_channels) + (length,) * spatial)
    mod = torch.randn(batch_size, mod_features)
    y = model(x, mod)

    assert y.ndim == x.ndim
    assert y.shape[0] == batch_size
    assert y.shape[1] == out_channels
    assert y.shape[2:] == x.shape[2:]

    ## Grads
    assert y.requires_grad

    loss = y.square().sum()
    loss.backward()

    for p in model.parameters():
        assert p.grad is not None
        assert torch.all(torch.isfinite(p.grad))

    # Save
    torch.save(model.state_dict(), tmp_path / "state.pth")

    # Load
    copy = make()
    copy.load_state_dict(torch.load(tmp_path / "state.pth", weights_only=True))

    model.eval()
    copy.eval()

    y = model(x, mod)
    y_copy = copy(x, mod)

    assert torch.allclose(y, y_copy)

    # Before 2.3, several CPU kernels lack a half precision implementation
    if TorchVersion(torch.__version__) < "2.3":
        return

    # Autocast
    with torch.autocast(device_type="cpu", dtype=torch.float16):
        y_auto = model(x, mod)

    assert y_auto.dtype == torch.float16
    assert torch.all(torch.isfinite(y_auto))

    err = (y - y_auto.to(torch.float32)).abs().flatten()

    assert torch.quantile(err, 0.99) < 2e-3
    assert torch.max(err) < 2e-2

    # Float16
    model.to(torch.float16)
    y16 = model(x.to(torch.float16), mod.to(torch.float16))

    model.to(torch.float32)
    y32 = model(x.to(torch.float32), mod.to(torch.float32))

    err = (y32 - y16).abs().flatten()

    assert torch.quantile(err, 0.99) < 2e-3
    assert torch.max(err) < 2e-2
