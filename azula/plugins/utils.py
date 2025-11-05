r"""Miscelaneous plugin helpers."""

__all__ = [
    "load_cards",
    "as_dtype",
]

import os
import sys
import torch
import yaml

from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
from typing import Dict, Optional


def as_dtype(name: Optional[str] = None) -> torch.dtype:
    if name is None:
        return None
    elif name == "float64":
        return torch.float64
    elif name == "float32":
        return torch.float32
    elif name == "float16":
        return torch.float16
    elif name == "bfloat16":
        return torch.bfloat16
    else:
        raise ValueError(f"Unknown data type '{name}'.")


def load_cards(plugin: ModuleType) -> Dict[str, SimpleNamespace]:
    r"""Returns the name-card mapping of pre-trained models available in a plugin.

    Arguments:
        plugin: The plugin module.

    Example:
        >>> cards = load_cards(azula.plugins.adm)
        >>> list(cards)
        ['imagenet_64x64_cond',
         'imagenet_128x128_cond',
         'imagenet_256x256',
         'imagenet_256x256_cond',
         'imagenet_512x512_cond',
         'ffhq_256x256']
    """

    if isinstance(plugin, str):
        plugin = sys.modules[plugin]

    file = os.path.join(os.path.dirname(plugin.__file__), "cards.yml")

    assert os.path.exists(file), f"{plugin} is not a plugin"

    with open(file, mode="r") as f:
        cards = yaml.safe_load(f)

    return {name: SimpleNamespace(**card) for name, card in cards.items()}


@contextmanager
def patch_diffusers():
    from tqdm import std
    from unittest.mock import patch

    with (
        patch("diffusers.utils.logging.tqdm_lib", std),
        patch("transformers.utils.logging.tqdm_lib", std),
        patch("transformers.modeling_utils.logger.warning_once"),
        patch("transformers.models.t5.tokenization_t5_fast.logger.warning_once"),
    ):
        yield
