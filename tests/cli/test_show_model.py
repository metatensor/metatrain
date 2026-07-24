import logging
import subprocess

import pytest
from metatomic.torch import ModelMetadata

from metatrain.cli.show import (
    _describe_metadata,
    _describe_target,
    _format_atomic_types,
    _target_type,
    show_model,
)
from metatrain.utils.data.target_info import (
    get_energy_target_info,
    get_generic_target_info,
)


@pytest.mark.parametrize("path_format", ["checkpoint", "checkpoint_url", "exported"])
def test_show(monkeypatch, tmp_path, caplog, path_format, MODEL_PATH_PET):
    """Test that show_model logs the model summary"""
    monkeypatch.chdir(tmp_path)

    if path_format == "checkpoint":
        path = str(MODEL_PATH_PET.with_suffix(".ckpt"))
    elif path_format == "checkpoint_url":
        path = f"file:{MODEL_PATH_PET.with_suffix('.ckpt')}"
    else:
        path = str(MODEL_PATH_PET)

    if path_format == "exported":
        expected = [
            "file type: exported model",
            "outputs:",
            "supported devices:",
        ]
    else:
        expected = [
            "file type: checkpoint",
            "architecture: pet",
            "model checkpoint version:",
            "targets:",
            "auxiliary outputs:",
        ]

    caplog.set_level(logging.INFO)
    show_model(path)

    for snippet in expected + ["energy:", "unit: eV", "atomic types:"]:
        assert snippet in caplog.text


def test_show_cli(monkeypatch, tmp_path, capfd, MODEL_PATH_PET):
    """Test that the show cli runs and prints the summary"""
    monkeypatch.chdir(tmp_path)

    subprocess.check_call(["mtt", "show", str(MODEL_PATH_PET.with_suffix(".ckpt"))])

    stdout = capfd.readouterr().out
    assert "architecture: pet" in stdout


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


def test_describe_target():
    """Test the description of a scalar target with gradients"""
    target_info = get_energy_target_info(
        "energy", {"unit": "eV"}, add_position_gradients=True
    )

    lines = _describe_target("energy", target_info)

    assert lines == [
        "  energy:",
        "    quantity: energy",
        "    unit: eV",
        "    type: scalar",
        "    sampled per: system",
        "    gradients: positions",
    ]


def test_formatting_helpers():
    """Test the metadata and atomic types formatting helpers"""
    metadata = ModelMetadata(
        name="test",
        description="A test model",
        authors=["John Doe", "Jane Smith"],
        references={"architecture": ["ref1"], "model": ["ref2"]},
    )
    assert _describe_metadata(metadata) == [
        "",
        "metadata:",
        "  name: test",
        "  description: A test model",
        "  authors: John Doe, Jane Smith",
        "  references:",
        "    - (architecture) ref1",
        "    - (model) ref2",
    ]
    assert _describe_metadata(ModelMetadata()) == []
    assert _format_atomic_types([1, 6, 999]) == "H (1), C (6), 999"
