r"""Tests for the azula.nn.unet module."""

import pytest
import torch

from pathlib import Path
from typing import Dict

from azula.nn.unet import UNet


@pytest.mark.parametrize("length", [15, 16])
@pytest.mark.parametrize("in_channels, out_channels", [(3, 5)])
@pytest.mark.parametrize("mod_features", [0, 16])
@pytest.mark.parametrize("attention_heads", [{}, {2: 1}])
@pytest.mark.parametrize("dropout", [None, 0.1])
@pytest.mark.parametrize("spatial", [1, 2])
@pytest.mark.parametrize("batch_size", [4])
def test_UNet(
    tmp_path: Path,
    length: int,
    in_channels: int,
    out_channels: int,
    mod_features: int,
    attention_heads: Dict[int, int],
    dropout: float,
    spatial: int,
    batch_size: int,
):
    if attention_heads and torch.__version__.startswith("1"):
        return

    make = lambda: UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        mod_features=mod_features,
        hid_channels=(5, 7, 11),
        hid_blocks=(1, 2, 3),
        attention_heads=attention_heads,
        dropout=dropout,
        spatial=spatial,
    )

    unet = make()
    unet.train()

    # Call
    x = torch.randn((batch_size, in_channels) + (length,) * spatial)
    mod = torch.randn(batch_size, mod_features)
    y = unet(x, mod)

    assert y.ndim == x.ndim
    assert y.shape[0] == batch_size
    assert y.shape[1] == out_channels
    assert y.shape[2:] == x.shape[2:]

    ## Grads
    assert y.requires_grad

    loss = y.square().sum()
    loss.backward()

    for p in unet.parameters():
        assert p.grad is not None
        assert torch.all(torch.isfinite(p.grad))

    # Save
    torch.save(unet.state_dict(), tmp_path / "state.pth")

    # Load
    copy = make()
    copy.load_state_dict(torch.load(tmp_path / "state.pth", weights_only=True))

    unet.eval()
    copy.eval()

    y_unet = unet(x, mod)
    y_copy = copy(x, mod)

    assert torch.allclose(y_unet, y_copy)
