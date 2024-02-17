import logging
import shutil
import subprocess
from pathlib import Path
import torch

import ase.io
import pytest
from omegaconf import OmegaConf

from metatensor.models.cli import eval_model
from metatensor.models.utils.model_io import load_model


RESOURCES_PATH = Path(__file__).parent.resolve() / ".." / "resources"
MODEL_PATH = RESOURCES_PATH / "bpnn-model.pt"
OPTIONS_PATH = RESOURCES_PATH / "eval.yaml"


@pytest.fixture
def model():
    return torch.jit.load(MODEL_PATH)


@pytest.fixture
def options():
    return OmegaConf.load(OPTIONS_PATH)


def test_eval_cli(monkeypatch, tmp_path):
    """Test succesful run of the eval script via the CLI with default arguments"""
    monkeypatch.chdir(tmp_path)
    shutil.copy(RESOURCES_PATH / "qm9_reduced_100.xyz", "qm9_reduced_100.xyz")

    command = [
        "metatensor-models",
        "eval",
        str(MODEL_PATH),
        str(OPTIONS_PATH),
    ]

    subprocess.check_call(command)

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
    assert "energy RMSE" in "".join([rec.message for rec in caplog.records])

    # Test file is written predictions
    frames = ase.io.read("foo.xyz", ":")
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
