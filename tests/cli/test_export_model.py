"""Test command line interface for the export functions.

Actual unit tests for the function are performed in `tests/utils/test_export`.
"""

import glob
import logging
import os
import shutil
import subprocess
from pathlib import Path

import huggingface_hub
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
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_export_cli(monkeypatch, tmp_path, output, dtype):
    """Test that the export cli runs without an error raise."""
    monkeypatch.chdir(tmp_path)

    dtype_string = str(dtype)[-2:]
    command = [
        "mtt",
        "export",
        str(RESOURCES_PATH / f"model-{dtype_string}-bit.ckpt"),
    ]

    if output is not None:
        command += ["-o", output]
    else:
        output = f"model-{dtype_string}-bit.pt"

    subprocess.check_call(command)
    assert Path(output).is_file()

    # Test if extensions are saved
    extensions_glob = glob.glob("extensions/")
    assert len(extensions_glob) == 1

    # Test that the model can be loaded
    model = load_model(output, extensions_directory="extensions/")

    # Check that the model has the correct dtype and is on cpu
    assert next(model.parameters()).dtype == dtype
    assert next(model.parameters()).device.type == "cpu"


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


def test_private_huggingface(monkeypatch, tmp_path):
    """Test that the export cli succeeds when exporting a private
    model from HuggingFace."""
    monkeypatch.chdir(tmp_path)

    HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN_METATRAIN")
    if HF_TOKEN is None:
        pytest.skip("HuggingFace token not found in environment.")
    assert len(HF_TOKEN) > 0

    huggingface_hub.upload_file(
        path_or_fileobj=str(RESOURCES_PATH / "model-32-bit.ckpt"),
        path_in_repo="model.ckpt",
        repo_id="metatensor/metatrain-test",
        commit_message="Overwrite test model with new version",
        token=HF_TOKEN,
    )

    command = [
        "mtt",
        "export",
        "https://huggingface.co/metatensor/metatrain-test/resolve/main/model.ckpt",
        f"--huggingface_api_token={HF_TOKEN}",
    ]

    output = "model.pt"

    subprocess.check_call(command)
    assert Path(output).is_file()

    # Test if extensions are saved
    extensions_glob = glob.glob("extensions/")
    assert len(extensions_glob) == 1

    # Test that the model can be loaded
    load_model(output, extensions_directory="extensions/")

    # also test with the token in the environment variable
    os.environ["HF_TOKEN"] = HF_TOKEN

    # remove output file and extensions
    os.remove(output)
    shutil.rmtree("extensions/")

    command = command[:-1]  # remove the token from the command line
    subprocess.check_call(command)
    assert Path(output).is_file()

    # Test if extensions are saved
    extensions_glob = glob.glob("extensions/")
    assert len(extensions_glob) == 1

    # Test that the model can be loaded
    load_model(output, extensions_directory="extensions/")


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
