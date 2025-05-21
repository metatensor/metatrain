"""Test command line interface for the export functions.

Actual unit tests for the function are performed in `tests/utils/test_export`.
"""

import glob
import logging
import os
import subprocess
from pathlib import Path
from subprocess import CalledProcessError

import pytest
import torch
from omegaconf import OmegaConf

from metatrain.cli.export import export_model
from metatrain.soap_bpnn import __model__
from metatrain.utils.architectures import find_all_architectures
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.io import load_model

from . import MODEL_HYPERS, RESOURCES_PATH


@pytest.mark.parametrize("path", [Path("exported.pt"), "exported.pt"])
def test_export(monkeypatch, tmp_path, path, caplog):
    """Tests the export_model function."""
    monkeypatch.chdir(tmp_path)
    caplog.set_level(logging.INFO)

    dataset_info = DatasetInfo(
        length_unit="angstrom",
        atomic_types={1},
        targets={"energy": get_energy_target_info({"unit": "eV"})},
    )
    model = __model__(model_hypers=MODEL_HYPERS, dataset_info=dataset_info)
    export_model(model, path)

    # Test if extensions are saved
    extensions_glob = glob.glob("extensions/")
    assert len(extensions_glob) == 1

    assert Path(path).is_file()

    # Test log message
    assert "Model exported to" in caplog.text


@pytest.mark.parametrize("output", [None, "exported.pt"])
@pytest.mark.parametrize("model_type", ["32-bit", "64-bit", "no-extensions"])
def test_export_cli(monkeypatch, tmp_path, output, model_type):
    """Test that the export cli runs without an error raise."""
    monkeypatch.chdir(tmp_path)

    command = [
        "mtt",
        "export",
        str(RESOURCES_PATH / f"model-{model_type}.ckpt"),
    ]

    if output is not None:
        command += ["-o", output]
    else:
        output = f"model-{model_type}.pt"

    subprocess.check_call(command)
    assert Path(output).is_file()

    # Test if extensions are saved
    extensions_glob = glob.glob("extensions/")
    if model_type == "no-extensions":
        assert len(extensions_glob) == 0
    else:
        assert len(extensions_glob) == 1

    # Test that the model can be loaded
    model = load_model(output, extensions_directory="extensions/")

    # Check that the model has the correct dtype and is on cpu
    if model_type == "32-bit":
        correct_dtype = torch.float32
    elif model_type == "64-bit":
        correct_dtype = torch.float64
    else:
        correct_dtype = torch.float64
    assert next(model.parameters()).dtype == correct_dtype
    assert next(model.parameters()).device.type == "cpu"


def test_export_with_env(monkeypatch, tmp_path):
    """Test that export with env variable works for local file."""
    monkeypatch.chdir(tmp_path)

    command = [
        "mtt",
        "export",
        str(RESOURCES_PATH / "model-32-bit.ckpt"),
    ]

    env = os.environ.copy()
    env["HF_TOKEN"] = "1234"

    subprocess.check_call(command, env=env)
    assert Path("model-32-bit.pt").is_file()


def test_export_cli_unknown_architecture(tmpdir):
    with tmpdir.as_cwd():
        torch.save({"architecture_name": "foo"}, "fake.ckpt")

        stdout = str(
            subprocess.run(["mtt", "export", "fake.ckpt"], capture_output=True).stdout
        )

        assert "architecture 'foo' not found in the available architectures" in stdout
        for architecture_name in find_all_architectures():
            assert architecture_name in stdout


def test_reexport(monkeypatch, tmp_path):
    """Test that an already exported model can be loaded and again exported."""
    monkeypatch.chdir(tmp_path)

    dataset_info = DatasetInfo(
        length_unit="angstrom",
        atomic_types={1, 6, 7, 8},
        targets={"energy": get_energy_target_info({"unit": "eV"})},
    )
    model = __model__(model_hypers=MODEL_HYPERS, dataset_info=dataset_info)

    export_model(model, "exported.pt")

    model_loaded = load_model("exported.pt")
    export_model(model_loaded, "exported_new.pt")

    assert Path("exported_new.pt").is_file()


def test_huggingface(monkeypatch, tmp_path):
    """Test that the export cli succeeds when exporting a private
    model from HuggingFace."""
    monkeypatch.chdir(tmp_path)

    HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN_METATRAIN")
    if HF_TOKEN is None:
        pytest.skip("HuggingFace token not found in environment.")
    assert len(HF_TOKEN) > 0

    command = [
        "mtt",
        "export",
        "https://huggingface.co/metatensor/metatrain-test/resolve/main/model.ckpt",
        f"--token={HF_TOKEN}",
    ]

    output = "model.pt"

    subprocess.check_call(command)
    assert Path(output).is_file()

    # Test if extensions are saved
    extensions_glob = glob.glob("extensions/")
    assert len(extensions_glob) == 1

    # Test that the model can be loaded
    load_model(output, extensions_directory="extensions/")


def test_huggingface_env(monkeypatch, tmp_path):
    """Test that huggingphase export works with env variable."""

    token = os.getenv("HUGGINGFACE_TOKEN_METATRAIN")
    if token is None:
        pytest.skip("HuggingFace token not found in environment.")

    monkeypatch.chdir(tmp_path)
    env = os.environ.copy()
    env["HF_TOKEN"] = token

    assert len(env["HF_TOKEN"]) > 0

    command = [
        "mtt",
        "export",
        "https://huggingface.co/metatensor/metatrain-test/resolve/main/model.ckpt",
    ]

    output = "model.pt"

    subprocess.check_call(command, env=env)
    assert Path(output).is_file()

    # Test if extensions are saved
    extensions_glob = glob.glob("extensions/")
    assert len(extensions_glob) == 1

    # Test that the model can be loaded
    load_model(output, extensions_directory="extensions/")


def test_token_env_error():
    command = [
        "mtt",
        "export",
        "https://huggingface.co/metatensor/metatrain-test/resolve/main/model.ckpt",
        "--token=1234",
    ]

    env = os.environ.copy()
    env["HF_TOKEN"] = "1234"

    with pytest.raises(CalledProcessError):
        subprocess.check_call(command, env=env)


def test_metadata(monkeypatch, tmp_path):
    """Test that the export cli does inject metadata."""
    monkeypatch.chdir(tmp_path)

    model_name = "test"
    conf = OmegaConf.create({"name": model_name})
    OmegaConf.save(config=conf, f="metadata.yaml")

    command = [
        "mtt",
        "export",
        str(RESOURCES_PATH / "model-32-bit.ckpt"),
        "--metadata=metadata.yaml",
    ]

    subprocess.check_call(command)
    model = load_model("model-32-bit.pt", extensions_directory="extensions/")

    assert f"This is the {model_name} model" in str(model.metadata())
