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
    "imagenet_512x512_xs": "https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-xs-2147483-0.200.pkl",
    "imagenet_512x512_s": "https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-s-2147483-0.190.pkl",
    "imagenet_512x512_m": "https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-m-2147483-0.155.pkl",
    "imagenet_512x512_l": "https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-l-1879048-0.155.pkl",
    "imagenet_512x512_xl": "https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-xl-1342177-0.155.pkl",
    "imagenet_512x512_xxl": "https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-xxl-0939524-0.150.pkl",
}
