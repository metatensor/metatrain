import os
import subprocess

import pytest
import yaml
from metatomic.torch import ModelMetadata

from metatrain.cli.info import (
    _metadata_info,
    _target_info,
    _target_type,
    info_model,
)
from metatrain.utils.data.target_info import (
    get_energy_target_info,
    get_generic_target_info,
)


@pytest.mark.parametrize("path_format", ["checkpoint", "checkpoint_url", "exported"])
def test_info(monkeypatch, tmp_path, capsys, path_format, MODEL_PATH_PET):
    """Test that info_model prints the model information as YAML"""
    monkeypatch.chdir(tmp_path)

    if path_format == "checkpoint":
        path = str(MODEL_PATH_PET.with_suffix(".ckpt"))
    elif path_format == "checkpoint_url":
        path = f"file:{MODEL_PATH_PET.with_suffix('.ckpt')}"
    else:
        path = str(MODEL_PATH_PET)

    info_model(path)
    info = yaml.safe_load(capsys.readouterr().out)

    assert info["length_unit"] == "angstrom"
    assert len(info["atomic_types"]) > 0
    assert info["parameters"] > 0

    if path_format == "exported":
        assert info["file_type"] == "exported model"
        assert "cpu" in info["supported_devices"]
        assert info["outputs"]["energy"]["unit"] == "eV"
    else:
        assert info["file_type"] == "checkpoint"
        assert info["architecture"]["name"] == "pet"
        assert "cutoff" in info["architecture"]["model"]
        assert "batch_size" in info["architecture"]["training"]
        assert "model_ckpt_version" in info["checkpoint"]
        assert info["targets"]["energy"]["unit"] == "eV"
        assert len(info["auxiliary_outputs"]) > 0


def test_info_huggingface(monkeypatch, tmp_path, capfd):
    """Test that the info cli works with a private model from HuggingFace"""
    monkeypatch.chdir(tmp_path)

    hf_token = os.getenv("HUGGINGFACE_TOKEN_METATRAIN")
    if hf_token is None or len(hf_token) == 0:
        pytest.skip("HuggingFace token not found in environment.")

    command = [
        "mtt",
        "info",
        "https://huggingface.co/metatensor/metatrain-test/resolve/main/model.ckpt",
        f"--token={hf_token}",
    ]
    subprocess.check_call(command)

    info = yaml.safe_load(capfd.readouterr().out)
    assert "architecture" in info


@pytest.mark.parametrize(
    "target_config, expected_type",
    [
        (
            {
                "quantity": "forces",
                "unit": "eV/A",
                "type": {"cartesian": {"rank": 1}},
                "sample_kind": "atom",
                "num_subtargets": 1,
            },
            "cartesian",
        ),
        (
            {
                "quantity": "",
                "unit": "",
                "type": {"spherical": {"irreps": [{"o3_lambda": 1, "o3_sigma": 1}]}},
                "sample_kind": "system",
                "num_subtargets": 1,
            },
            "spherical",
        ),
    ],
)
def test_target_type(target_config, expected_type):
    """Test the type name for non-scalar targets"""
    target_info = get_generic_target_info("mtt::my_target", target_config)
    assert _target_type(target_info) == expected_type


def test_target_info():
    """Test the information of a scalar target with gradients"""
    target_info = get_energy_target_info(
        "energy", {"unit": "eV"}, add_position_gradients=True
    )

    assert _target_info(target_info) == {
        "quantity": "energy",
        "unit": "eV",
        "type": "scalar",
        "sample_kind": "system",
        "gradients": ["positions"],
    }


def test_metadata_info():
    """Test the metadata information"""
    metadata = ModelMetadata(
        name="test",
        description="A test model",
        authors=["John Doe", "Jane Smith"],
        references={"architecture": ["ref1"], "model": ["ref2"]},
    )
    assert _metadata_info(metadata) == {
        "name": "test",
        "description": "A test model",
        "authors": ["John Doe", "Jane Smith"],
        "references": {"architecture": ["ref1"], "model": ["ref2"]},
    }
    assert _metadata_info(ModelMetadata()) == {}
