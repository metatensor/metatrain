from pathlib import Path

import pytest
import torch
from metatensor.torch.atomistic import MetatensorAtomisticModel

from metatrain.experimental.soap_bpnn.model import SoapBpnn
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
    if str(path).startswith("file:"):
        # test that the checkpoint is also copied to the current directory
        assert Path("model-32-bit.ckpt").exists()


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


def test_load_model_unknown_model(tmpdir):
    architecture_name = "experimental.soap_bpnn"
    path = "fake.ckpt"

    with tmpdir.as_cwd():
        torch.save({"architecture_name": architecture_name}, path)

        match = (
            f"path '{path}' is not a valid checkpoint for the {architecture_name} "
            "architecture"
        )
        with pytest.raises(ValueError, match=match):
            load_model(path, architecture_name=architecture_name)


def test_load_model_no_architecture_name(monkeypatch, tmpdir):
    monkeypatch.chdir(tmpdir)
    architecture_name = "experimental.soap_bpnn"
    path = "fake.ckpt"
    torch.save({"not_architecture_name": architecture_name}, path)

    match = "No architecture name found in the checkpoint"
    with pytest.raises(ValueError, match=match):
        load_model(path, architecture_name=architecture_name)
