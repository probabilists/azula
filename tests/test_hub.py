r"""Tests for the azula.hub module."""

import os
import pytest

from azula.hub import download, get_dir, set_dir


def test_default_dir():
    default_dir = get_dir()

    assert isinstance(default_dir, str)

    os.makedirs(default_dir, exist_ok=True)


def test_set_dir(tmp_path):
    set_dir(tmp_path)
    cache_dir = get_dir()

    assert isinstance(cache_dir, str)
    assert os.path.samefile(cache_dir, tmp_path)


def test_download(tmp_path):
    # Set cache dir
    set_dir(tmp_path)

    # With filename
    download(
        url="https://raw.githubusercontent.com/probabilists/azula/master/LICENSE",
        filename=tmp_path / "LICENSE",
    )

    with open(tmp_path / "LICENSE") as f:
        text = f.read()

        assert "MIT License" in text
        assert "The Probabilists" in text

    # Without filename
    filename = download(
        url="https://raw.githubusercontent.com/probabilists/azula/master/LICENSE",
    )

    assert os.path.samefile(os.path.dirname(filename), tmp_path)

    with open(filename) as f:
        text = f.read()

        assert "MIT License" in text
        assert "The Probabilists" in text

    # Hash prefix
    download(
        url="https://raw.githubusercontent.com/probabilists/azula/master/LICENSE",
        hash_prefix="sha256:c8adb00fadb8f4bf",
    )

    with pytest.raises(AssertionError):
        download(
            url="https://raw.githubusercontent.com/probabilists/azula/master/LICENSE",
            hash_prefix="sha256:abcdefghijklmnop",
        )

    # TODO (francois-rozet)
    # Find a URL to test Google Drive download
