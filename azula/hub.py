r"""Utilities for downloading models."""

__all__ = [
    "get_dir",
    "set_dir",
    "download",
]

import gdown
import os
import re
import sys
import torch

from typing import Optional

AZULA_HUB: str = os.path.expanduser("~/.cache/azula/hub")


def get_dir() -> str:
    r"""Returns the cache directory used for storing models & weights."""

    return AZULA_HUB


def set_dir(cache_dir: str):
    r"""Sets the cache directory used for storing models & weights."""

    global AZULA_HUB

    cache_dir = os.path.expanduser(cache_dir)
    cache_dir = os.path.abspath(cache_dir)

    AZULA_HUB = cache_dir


def download(
    url: str,
    filename: Optional[str] = None,
    quiet: bool = False,
) -> str:
    r"""Downloads data at a given URL to a local file.

    If a file with the same name exists, the download is skipped.

    Arguments:
        url: A URL. Google Drive URLs are supported.
        filename: A local file name. If :py:`None`, use the sanitized URL instead.
        quiet: Whether to keep it quiet in the terminal or not.
    """

    if filename is None:
        filename = re.sub("[ /\\\\|?%*:\"'<>]", "", url)
        filename = os.path.join(get_dir(), filename)
    else:
        filename = os.path.expanduser(filename)
        filename = os.path.abspath(filename)

    if os.path.exists(filename):
        if not quiet:
            print(f"Skipping download as {filename} already exists.", file=sys.stderr)
    else:
        if not quiet:
            print(f"Downloading {url} to {filename}", file=sys.stderr)

        if "drive.google" in url:
            gdown.download(url, filename, quiet=quiet)
        else:
            torch.hub.download_url_to_file(url, filename, progress=not quiet)

    return filename
