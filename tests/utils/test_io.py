import os
from pathlib import Path

import pytest
import torch
from metatomic.torch import AtomisticModel

from metatrain.soap_bpnn.model import SoapBpnn
from metatrain.utils.io import (
    check_file_extension,
    is_exported_file,
    load_model,
    model_from_checkpoint,
    trainer_from_checkpoint,
)

from . import RESOURCES_PATH


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

    # TODO: test that weights are the expected if loading with `context == 'export'`.
    # One can use `list(model.bpnn[0].parameters())[0][0]` to get some weights. But,
    # currently weights of the `"export"` and the `"restart"` context are the same...


def test_load_model_checkpoint_wrong_version(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    path = RESOURCES_PATH / "model-64-bit.ckpt"
    model = torch.load(path, weights_only=False, map_location="cpu")
    model["model_ckpt_version"] = 5000000

    file = "model-64-bit-version5000000.ckpt"
    torch.save(model, file)

    message = (
        "Unable to load the model checkpoint from 'model-64-bit-version5000000.ckpt' "
        "for the 'soap_bpnn' architecture: the checkpoint is using version 5000000, "
        "while the current version is 1; and trying to upgrade the checkpoint failed."
    )
    with pytest.raises(RuntimeError, match=message):
        model_from_checkpoint(file)


def test_load_trainer_checkpoint_wrong_version(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    path = RESOURCES_PATH / "model-64-bit.ckpt"
    model = torch.load(path, weights_only=False, map_location="cpu")
    model["trainer_ckpt_version"] = 5000000

    file = "model-64-bit-version5000000.ckpt"
    torch.save(model, file)

    message = (
        "Unable to load the trainer checkpoint from 'model-64-bit-version5000000.ckpt' "
        "for the 'soap_bpnn' architecture: the checkpoint is using version 5000000, "
        "while the current version is 1; and trying to upgrade the checkpoint failed."
    )
    with pytest.raises(RuntimeError, match=message):
        trainer_from_checkpoint(file, "restart", hypers={})


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
    assert type(model) is AtomisticModel


@pytest.mark.parametrize("suffix", [".yml", ".yaml"])
def test_load_model_yaml(suffix):
    match = f"path 'foo{suffix}' seems to be a YAML option file and not a model"
    with pytest.raises(ValueError, match=match):
        load_model(f"foo{suffix}")


def test_load_model_token():
    """Test that the export cli succeeds when exporting a private
    model from HuggingFace."""

    hf_token = os.getenv("HUGGINGFACE_TOKEN_METATRAIN")
    if hf_token is None or len(hf_token) == 0:
        pytest.skip("HuggingFace token not found in environment.")

    path = "https://huggingface.co/metatensor/metatrain-test/resolve/main/model.ckpt"
    load_model(path, hf_token=hf_token)


def test_load_model_token_invalid_url_style():
    hf_token = os.getenv("HUGGINGFACE_TOKEN_METATRAIN")
    if hf_token is None or len(hf_token) == 0:
        pytest.skip("HuggingFace token not found in environment.")

    # change `resolve` to ``foo`` to make the URL scheme invalid
    path = "https://huggingface.co/metatensor/metatrain-test/foo/main/model.ckpt"

    with pytest.raises(
        ValueError,
        match=f"URL '{path}' has an invalid format for the Hugging Face Hub.",
    ):
        load_model(path, hf_token=hf_token)
