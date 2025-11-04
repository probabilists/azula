r"""Tests for the azula.nn.vit module."""

import pytest
import torch

from pathlib import Path

from azula.nn.vit import ViT


@pytest.mark.parametrize("length", [16])
@pytest.mark.parametrize("in_channels, out_channels", [(3, 5)])
@pytest.mark.parametrize("mod_features", [0, 16])
@pytest.mark.parametrize("attention_heads", [4])
@pytest.mark.parametrize("dropout", [None, 0.1])
@pytest.mark.parametrize("spatial", [1, 2])
@pytest.mark.parametrize("rope", [False, True])
@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("batch_size", [4])
def test_ViT(
    tmp_path: Path,
    length: int,
    in_channels: int,
    out_channels: int,
    mod_features: int,
    attention_heads: int,
    dropout: float,
    spatial: int,
    rope: bool,
    checkpointing: bool,
    batch_size: int,
):
    if torch.__version__.startswith("1"):
        return

    make = lambda: ViT(
        in_channels=in_channels,
        out_channels=out_channels,
        mod_features=mod_features,
        hid_channels=16,
        hid_blocks=3,
        attention_heads=attention_heads,
        dropout=dropout,
        spatial=spatial,
        patch_size=[4] * spatial,
        rope=rope,
        checkpointing=checkpointing,
    )

    vit = make()
    vit.train()

    # Call
    x = torch.randn((batch_size, in_channels) + (length,) * spatial)
    mod = torch.randn(batch_size, mod_features)
    y = vit(x, mod)

    assert y.ndim == x.ndim
    assert y.shape[0] == batch_size
    assert y.shape[1] == out_channels
    assert y.shape[2:] == x.shape[2:]

    ## Grads
    assert y.requires_grad

    loss = y.square().sum()
    loss.backward()

    for p in vit.parameters():
        assert p.grad is not None
        assert torch.all(torch.isfinite(p.grad))

    # Save
    torch.save(vit.state_dict(), tmp_path / "state.pth")

    # Load
    copy = make()
    copy.load_state_dict(torch.load(tmp_path / "state.pth", weights_only=True))

    vit.eval()
    copy.eval()

    y_vit = vit(x, mod)
    y_copy = copy(x, mod)

    assert torch.allclose(y_vit, y_copy)
