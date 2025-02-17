r"""Pre-trained models database."""

from typing import Dict, List, Tuple


def get(key: str) -> Tuple[str, str, Dict]:
    r"""Returns the URL, hash and config of a pre-trained model.

    Arguments:
        key: The pre-trained model key.
    """

    return URLS[key], HASHES[key], CONFIGS[key]


def keys() -> List[str]:
    r"""Returns the list of available pre-trained models."""

    return list(URLS.keys())


URLS = {
    "danbooru_128x128": "https://the-eye.eu/public/AI/models/v-diffusion/danbooru_128.pth",
    "imagenet_128x128": "https://the-eye.eu/public/AI/models/v-diffusion/imagenet_128.pth",
    "wikiart_128x128": "https://the-eye.eu/public/AI/models/v-diffusion/wikiart_128.pth",
    "wikiart_256x256": "https://the-eye.eu/public/AI/models/v-diffusion/wikiart_256.pth",
    "yfcc_512x512": "https://the-eye.eu/public/AI/models/v-diffusion/yfcc_1.pth",
    "yfcc_512x512_large": "https://the-eye.eu/public/AI/models/v-diffusion/yfcc_2.pth",
}

HASHES = {
    "danbooru_128x128": "sha256:1728940d3531504246dbdc75748205fd8a24238a17e90feb82a64d7c8078c449",
    "imagenet_128x128": "sha256:cac117cd0ed80390b2ae7f3d48bf226fd8ee0799d3262c13439517da7c214a67",
    "wikiart_128x128": "sha256:b3ca8d0cf8bd47dcbf92863d0ab6e90e5be3999ab176b294c093431abdce19c1",
    "wikiart_256x256": "sha256:da45c38aa31cd0d2680d29a3aaf2f50537a4146d80bba2ca3e7a18d227d9b627",
    "yfcc_512x512": "sha256:a1c0f6baaf89cb4c461f691c2505e451ff1f9524744ce15332b7987cc6e3f0c8",
    "yfcc_512x512_large": "sha256:69ad4e534feaaebfd4ccefbf03853d5834231ae1b5402b9d2c3e2b331de27907",
}

CONFIGS = {
    "danbooru_128x128": {"key": "danbooru_128"},
    "imagenet_128x128": {"key": "imagenet_128"},
    "wikiart_128x128": {"key": "wikiart_128"},
    "wikiart_256x256": {"key": "wikiart_256"},
    "yfcc_512x512": {"key": "yfcc_1"},
    "yfcc_512x512_large": {"key": "yfcc_2"},
}
