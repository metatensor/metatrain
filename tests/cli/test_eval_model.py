import logging
import shutil
import subprocess
from pathlib import Path

import ase.io
import pytest
import torch
from omegaconf import OmegaConf

from metatensor.models.cli.eval import eval_model


RESOURCES_PATH = Path(__file__).parent.resolve() / ".." / "resources"
MODEL_PATH = RESOURCES_PATH / "bpnn-model.pt"
OPTIONS_PATH = RESOURCES_PATH / "eval.yaml"


@pytest.fixture
def model():
    return torch.jit.load(MODEL_PATH)


@pytest.fixture
def options():
    return OmegaConf.load(OPTIONS_PATH)


def test_eval_cli(monkeypatch, tmp_path, capsys):
    """Test succesful run of the eval script via the CLI with default arguments"""
    monkeypatch.chdir(tmp_path)
    shutil.copy(RESOURCES_PATH / "qm9_reduced_100.xyz", "qm9_reduced_100.xyz")

    command = [
        "metatensor-models",
        "eval",
        str(MODEL_PATH),
        str(OPTIONS_PATH),
    ]

    output = subprocess.check_output(command, stderr=subprocess.STDOUT)

    assert b"energy RMSE" in output

    assert Path("output.xyz").is_file()


def test_eval(monkeypatch, tmp_path, caplog, model, options):
    """Test that eval via python API runs without an error raise."""
    monkeypatch.chdir(tmp_path)
    caplog.set_level(logging.INFO)

    shutil.copy(RESOURCES_PATH / "qm9_reduced_100.xyz", "qm9_reduced_100.xyz")

    eval_model(
        model=model,
        options=options,
        output="foo.xyz",
    )

    # Test target predictions
    log = "".join([rec.message for rec in caplog.records])
    assert "energy RMSE" in log
    assert "dataset with index" not in log

    # Test file is written predictions
    frames = ase.io.read("foo.xyz", ":")
    frames[0].info["energy"]


def test_eval_multi_dataset(monkeypatch, tmp_path, caplog, model, options):
    """Test that eval runs for multiple datasets should be evaluated."""
    monkeypatch.chdir(tmp_path)
    caplog.set_level(logging.INFO)

    shutil.copy(RESOURCES_PATH / "qm9_reduced_100.xyz", "qm9_reduced_100.xyz")

    eval_model(
        model=model,
        options=OmegaConf.create([options, options]),
        output="foo.xyz",
    )

    # Test target predictions
    log = "".join([rec.message for rec in caplog.records])
    assert "index 0" in log
    assert "index 1" in log

    # Test file is written predictions
    for i in range(2):
        frames = ase.io.read(f"foo_{i}.xyz", ":")
        frames[0].info["energy"]


def test_eval_no_targets(monkeypatch, tmp_path, model, options):
    monkeypatch.chdir(tmp_path)

    shutil.copy(RESOURCES_PATH / "qm9_reduced_100.xyz", "qm9_reduced_100.xyz")

    options.pop("targets")

    eval_model(
        model=model,
        options=options,
    )

    assert Path("output.xyz").is_file()
