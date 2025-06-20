r"""Miscelaneous plugin helpers."""

__all__ = [
    "load_cards",
]

import os
import sys
import yaml

from types import ModuleType, SimpleNamespace
from typing import Dict


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
