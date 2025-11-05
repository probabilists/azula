r"""Miscelaneous plugin helpers."""

__all__ = [
    "load_cards",
]

import os
import sys
import torch
import yaml

from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
from typing import Dict, Optional


def as_torch_dtype(name: Optional[str] = None) -> torch.dtype:
    if name is None:
        return None

    dtype = getattr(torch, name, None)

    if isinstance(dtype, torch.dtype):
        return dtype
    else:
        raise ValueError(f"Unknown data type '{name}'.")


def load_cards(plugin: ModuleType) -> Dict[str, SimpleNamespace]:
    r"""Returns the name-card mapping of pre-trained models available in a plugin.

    Arguments:
        plugin: The plugin module.

    Example:
        >>> cards = load_cards(azula.plugins.adm)
        >>> list(cards.keys())
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

    for card in cards.values():
        if "dtype_map" in card:
            card["dtype_map"] = {k: as_torch_dtype(v) for k, v in card["dtype_map"].items()}

    return {name: SimpleNamespace(**card) for name, card in cards.items()}


@contextmanager
def patch_diffusers():
    from safetensors.torch import load, load_file
    from tqdm import std
    from unittest.mock import patch

    def monkey_load(checkpoint_file, *args, **kwargs):
        if isinstance(checkpoint_file, dict):
            return checkpoint_file
        elif kwargs.get("map_location", "cpu") == "meta":
            return load_file(checkpoint_file)
        else:
            with open(checkpoint_file, "rb") as f:
                return load(f.read())

    with (
        patch("diffusers.utils.logging.tqdm_lib", std),
        patch("diffusers.models.model_loading_utils.load_state_dict", monkey_load),
        patch("diffusers.models.modeling_utils.load_state_dict", monkey_load),
        patch("diffusers.loaders.single_file_utils.load_state_dict", monkey_load),
        patch("diffusers.loaders.unet.load_state_dict", monkey_load),
        patch("transformers.utils.logging.tqdm_lib", std),
        patch("transformers.modeling_utils.logger.warning_once"),
        patch("transformers.models.t5.tokenization_t5_fast.logger.warning_once"),
    ):
        yield


def patch_mmap_safetensors(*modules):
    for m in modules:
        for p in m.parameters():
            p.data = p.data.clone()

        for b in m.buffers():
            b.data = b.data.clone()
