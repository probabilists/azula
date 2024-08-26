r"""Pre-trained models database."""

from typing import Dict, List, Tuple


def get(key: str) -> Tuple[str, Dict]:
    r"""Returns the URL and config of a pre-trained model.

    Arguments:
        key: The pre-trained model key.
    """

    return URLS[key], CONFIGS[key]


def keys() -> List[str]:
    r"""Returns the list of available pre-trained models."""

    return list(URLS.keys())


URLS = {
    "imagenet_64x64": "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_diffusion.pt",
    "imagenet_256x256": "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion.pt",
    "imagenet_256x256_uncond": "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt",
    "ffhq_256x256": "https://drive.google.com/uc?id=1BGwhRWUoguF-D8wlZ65tf227gp3cDUDh",
}

CONFIGS = {
    "imagenet_64x64": {
        "schedule_name": "cosine",
        "attention_resolutions": {32, 16, 8},
        "channel_mult": (1, 2, 3, 4),
        "dropout": 0.1,
        "image_size": 64,
        "num_channels": 192,
        "num_classes": 1000,
        "num_head_channels": 64,
        "num_res_blocks": 3,
        "use_new_attention_order": True,
    },
    "imagenet_256x256": {
        "attention_resolutions": {32, 16, 8},
        "channel_mult": (1, 1, 2, 2, 4, 4),
        "image_size": 256,
        "num_channels": 256,
        "num_classes": 1000,
        "num_head_channels": 64,
        "num_res_blocks": 2,
    },
    "imagenet_256x256_uncond": {
        "attention_resolutions": {32, 16, 8},
        "channel_mult": (1, 1, 2, 2, 4, 4),
        "image_size": 256,
        "num_channels": 256,
        "num_classes": None,
        "num_head_channels": 64,
        "num_res_blocks": 2,
    },
    "ffhq_256x256": {
        "attention_resolutions": {16},
        "channel_mult": (1, 1, 2, 2, 4, 4),
        "image_size": 256,
        "num_channels": 128,
        "num_classes": None,
        "num_heads": 4,
        "num_head_channels": 64,
        "num_res_blocks": 1,
    },
}
