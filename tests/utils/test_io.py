import os
from pathlib import Path

import pytest
from metatensor.torch.atomistic import MetatensorAtomisticModel

from metatrain.soap_bpnn.model import SoapBpnn
from metatrain.utils.io import check_file_extension, is_exported_file, load_model

from . import RESOURCES_PATH


def is_None(*args, **kwargs) -> None:
    return None


@pytest.mark.parametrize("filename", ["example.txt", Path("example.txt")])
def test_check_suffix(filename):
    result = check_file_extension(filename, ".txt")

    assert str(result) == "example.txt"
    assert isinstance(result, type(filename))


@pytest.mark.parametrize("filename", ["example", Path("example")])
def test_warning_on_missing_suffix(filename):
    match = r"The file name should have a '\.txt' file extension."
    with pytest.warns(UserWarning, match=match):
        result = check_file_extension(filename, ".txt")

    assert str(result) == "example.txt"
    assert isinstance(result, type(filename))


def test_is_exported_file():
    assert is_exported_file(RESOURCES_PATH / "model-32-bit.pt")
    assert not is_exported_file(RESOURCES_PATH / "model-32-bit.ckpt")


@pytest.mark.parametrize(
    "path",
    [
        RESOURCES_PATH / "model-32-bit.ckpt",
        str(RESOURCES_PATH / "model-32-bit.ckpt"),
        f"file:{str(RESOURCES_PATH / 'model-32-bit.ckpt')}",
    ],
)
def test_load_model_checkpoint(path):
    model = load_model(path)
    assert type(model) is SoapBpnn


@pytest.mark.parametrize(
    "path",
    [
        RESOURCES_PATH / "model-32-bit.pt",
        str(RESOURCES_PATH / "model-32-bit.pt"),
        f"file:{str(RESOURCES_PATH / 'model-32-bit.pt')}",
    ],
)
def test_load_model_exported(path):
    model = load_model(path)
    assert type(model) is MetatensorAtomisticModel


@pytest.mark.parametrize("suffix", [".yml", ".yaml"])
def test_load_model_yaml(suffix):
    match = f"path 'foo{suffix}' seems to be a YAML option file and not a model"
    with pytest.raises(ValueError, match=match):
        load_model(f"foo{suffix}")


def test_load_model_token():
    """Test that the export cli succeeds when exporting a private
    model from HuggingFace."""

    token = os.getenv("HUGGINGFACE_TOKEN_METATRAIN")
    if token is None:
        pytest.skip("HuggingFace token not found in environment.")
    assert len(token) > 0

    path = "https://huggingface.co/metatensor/metatrain-test/resolve/main/model.ckpt"
    load_model(path, token=token)


def test_load_model_token_invalid_url_style():
    token = os.getenv("HUGGINGFACE_TOKEN_METATRAIN")
    if token is None:
        pytest.skip("HuggingFace token not found in environment.")
    assert len(token) > 0

    # change `resolve` to ``foo`` to make the URL scheme invalid
    path = "https://huggingface.co/metatensor/metatrain-test/foo/main/model.ckpt"

    with pytest.raises(
        ValueError,
        match=f"URL '{path}' has an invalid format for the Hugging Face Hub.",
    ):
        load_model(path, token=token)
