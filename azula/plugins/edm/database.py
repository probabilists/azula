r"""Pre-trained models database."""

from typing import List


def get(key: str) -> str:
    r"""Returns the URL of a pre-trained model.

    Arguments:
        key: The pre-trained model key.
    """

    return URLS[key]


def keys() -> List[str]:
    r"""Returns the list of available pre-trained models."""

    return list(URLS.keys())


URLS = {
    "cifar10_32x32": "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-ve.pkl",
    "cifar10_32x32_cond": "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-ve.pkl",
    "afhq_64x64": "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-afhqv2-64x64-uncond-ve.pkl",
    "ffhq_64x64": "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-ffhq-64x64-uncond-ve.pkl",
    "imagenet_64x64_cond": "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl",
}
