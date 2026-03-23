r"""Tests for the azula.nn.vit module."""

import pytest
import torch

from pathlib import Path

from azula.nn.dit import DiT


@pytest.mark.parametrize("length", [16])
@pytest.mark.parametrize("in_channels, out_channels", [(3, 5)])
@pytest.mark.parametrize("mod_features", [0, 16])
@pytest.mark.parametrize("pos_channels", [1, 2])
@pytest.mark.parametrize("attention_heads", [4])
@pytest.mark.parametrize("dropout", [None, 0.1])
@pytest.mark.parametrize("rope", [False, True])
@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("batch_size", [4])
def test_DiT(
    tmp_path: Path,
    length: int,
    in_channels: int,
    out_channels: int,
    mod_features: int,
    pos_channels: int,
    attention_heads: int,
    dropout: float,
    rope: bool,
    checkpointing: bool,
    batch_size: int,
) -> None:
    make = lambda: DiT(
        in_channels=in_channels,
        out_channels=out_channels,
        mod_features=mod_features,
        pos_channels=pos_channels,
        hid_channels=16,
        hid_blocks=3,
        attention_heads=attention_heads,
        dropout=dropout,
        rope=rope,
        checkpointing=checkpointing,
    )

    dit = make()
    dit.train()

    # Call
    x = torch.randn(batch_size, length, in_channels)
    mod = torch.randn(batch_size, mod_features)
    pos = torch.randn(length, pos_channels)
    y = dit(x, mod, pos=pos)

    assert y.ndim == x.ndim
    assert y.shape == (batch_size, length, out_channels)

    ## Grads
    assert y.requires_grad

    loss = y.square().sum()
    loss.backward()

    for p in dit.parameters():
        assert p.grad is not None
        assert torch.all(torch.isfinite(p.grad))

    # Save
    torch.save(dit.state_dict(), tmp_path / "state.pth")

    # Load
    copy = make()
    copy.load_state_dict(torch.load(tmp_path / "state.pth", weights_only=True))

    dit.eval()
    copy.eval()

    y = dit(x, mod, pos=pos)
    y_copy = copy(x, mod, pos=pos)

    assert torch.allclose(y, y_copy)

    # Float16
    if torch.__version__ < "2.3":
        return

    dit.to(torch.float16)
    y16 = dit(x.to(torch.float16), mod.to(torch.float16), pos=pos.to(torch.float16))

    dit.to(torch.float32)
    y32 = dit(x.to(torch.float32), mod.to(torch.float32), pos=pos.to(torch.float32))

    err = (y32 - y16).abs().flatten()

    assert torch.quantile(err, 0.99) < 1e-3
    assert torch.max(err) < 1e-2
