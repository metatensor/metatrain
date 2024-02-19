import glob
import shutil
import subprocess
import warnings
from pathlib import Path

import ase.io
import metatensor.torch  # noqa
import pytest
import torch
from omegaconf import OmegaConf
from omegaconf.errors import ConfigKeyError

from metatensor.models.cli.train_model import check_architecture_name, train_model


RESOURCES_PATH = Path(__file__).parent.resolve() / ".." / "resources"
DATASET_PATH = RESOURCES_PATH / "qm9_reduced_100.xyz"
OPTIONS_PATH = RESOURCES_PATH / "options.yaml"
MODEL_PATH = RESOURCES_PATH / "bpnn-model.ckpt"


@pytest.fixture
def options():
    return OmegaConf.load(OPTIONS_PATH)


@pytest.mark.parametrize("output", [None, "mymodel.pt"])
def test_train(monkeypatch, tmp_path, output):
    """Test that training via the training cli runs without an error raise."""
    monkeypatch.chdir(tmp_path)
    shutil.copy(DATASET_PATH, "qm9_reduced_100.xyz")
    shutil.copy(OPTIONS_PATH, "options.yaml")

    command = ["metatensor-models", "train", "options.yaml"]

    if output is not None:
        command += ["-o", output]
    else:
        output = "model.pt"

    subprocess.check_call(command)
    assert Path(output).is_file()

    # Test if fully expanded options.yaml file is written
    assert len(glob.glob("outputs/*/*/options.yaml")) == 1

    # Test if logfile is written
    assert len(glob.glob("outputs/*/*/train.log")) == 1

    # Open the log file and check if the logging is correct
    with open(glob.glob("outputs/*/*/train.log")[0]) as f:
        log = f.read()

    assert "This log is also available"
    assert "[INFO]" in log
    assert "Epoch" in log
    assert "loss" in log
    assert "validation" in log
    assert "train" in log
    assert "energy" in log


@pytest.mark.parametrize("test_set_file", (True, False))
@pytest.mark.parametrize("validation_set_file", (True, False))
def test_train_explicit_validation_test(
    monkeypatch, tmp_path, test_set_file, validation_set_file, options
):
    """Test that training via the training cli runs without an error raise
    also when the validation and test sets are provided explicitly."""
    monkeypatch.chdir(tmp_path)

    structures = ase.io.read(DATASET_PATH, ":")

    ase.io.write("qm9_reduced_100.xyz", structures[:50])

    if test_set_file:
        ase.io.write("test.xyz", structures[50:80])
        options["validation_set"] = options["training_set"].copy()
        options["validation_set"]["structures"]["read_from"] = "test.xyz"

    if validation_set_file:
        ase.io.write("validation.xyz", structures[80:])
        options["test_set"] = options["training_set"].copy()
        options["test_set"]["structures"]["read_from"] = "validation.xyz"

    train_model(options)

    assert Path("model.pt").is_file()


def test_continue(options, monkeypatch, tmp_path):
    """Test that continuing training from a checkpoint runs without an error raise."""
    monkeypatch.chdir(tmp_path)
    shutil.copy(DATASET_PATH, "qm9_reduced_100.xyz")

    train_model(options, continue_from=MODEL_PATH)


def test_continue_different_dataset(options, monkeypatch, tmp_path):
    """Test that continuing training from a checkpoint runs without an error raise
    with a different dataset than the original."""
    monkeypatch.chdir(tmp_path)
    shutil.copy(RESOURCES_PATH / "ethanol_reduced_100.xyz", "ethanol_reduced_100.xyz")

    options["training_set"]["structures"]["read_from"] = "ethanol_reduced_100.xyz"
    options["training_set"]["targets"]["energy"]["key"] = "energy"

    train_model(options, continue_from=MODEL_PATH)


def test_continue_from_exported(options, monkeypatch, tmp_path):
    """Test that continuing training from an exported model raises an error."""
    monkeypatch.chdir(tmp_path)
    shutil.copy(DATASET_PATH, "qm9_reduced_100.xyz")

    # check that this warns and then errors out
    with pytest.raises(SystemExit):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")  # turn all warnings into catchable events
            train_model(options, continue_from=RESOURCES_PATH / "bpnn-model.pt")
            assert any(
                "Please use a .ckpt (checkpoint) file instead" in str(warning.message)
                for warning in w
            )


def test_hydra_arguments():
    """Test if hydra arguments work."""
    option_path = str(RESOURCES_PATH / "options.yaml")
    out = subprocess.check_output(
        ["metatensor-models", "train", option_path, "--hydra=--help"]
    )
    # Check that num_epochs is override is succesful
    assert "num_epochs: 1" in str(out)


def test_no_architecture_name(options):
    """Test error raise if architecture.name is not set."""
    options["architecture"].pop("name")

    with pytest.raises(ConfigKeyError, match="Architecture name is not defined!"):
        train_model(options)


@pytest.mark.parametrize("seed", [1234, None, 0, -123])
@pytest.mark.parametrize("architecture_name", ["experimental.soap_bpnn"])
def test_model_consistency_with_seed(
    options, monkeypatch, tmp_path, architecture_name, seed, capsys
):
    """Checks final model consistency with a fixed seed."""
    monkeypatch.chdir(tmp_path)
    shutil.copy(DATASET_PATH, "qm9_reduced_100.xyz")

    options["architecture"]["name"] = architecture_name
    options["seed"] = seed

    if seed is not None and seed < 0:
        with pytest.raises(SystemExit):
            train_model(options)
            captured = capsys.readouterr()
            assert "should be a positive number or None." in captured.out
        return

    train_model(options, output="model1.pt")
    train_model(options, output="model2.pt")

    m1 = torch.load("model1.ckpt")
    m2 = torch.load("model2.ckpt")

    for index, i in enumerate(m1["model_state_dict"]):
        tensor1 = m1["model_state_dict"][i]
        tensor2 = m2["model_state_dict"][i]

        # The first tensor only depend on the chemical compositions (not on the
        # seed) and should alwyas be the same.
        if index == 0:
            assert torch.allclose(tensor1, tensor2)
        else:
            if seed is None:
                assert not torch.allclose(tensor1, tensor2)
            else:
                assert torch.allclose(tensor1, tensor2)


def test_error_base_precision(options, monkeypatch, tmp_path, capsys):
    """Test unsupported `base_precision`"""
    monkeypatch.chdir(tmp_path)

    options["base_precision"] = "123"

    with pytest.raises(SystemExit):
        train_model(options)
        captured = capsys.readouterr()
        assert "Only 64, 32 or 16 are possible values for" in captured.out


def test_check_architecture_name():
    check_architecture_name("experimental.soap_bpnn")


def test_check_architecture_name_suggest():
    name = "soap-bpnn"
    match = f"Architecture {name!r} is not a valid architecture."
    with pytest.raises(ValueError, match=match):
        check_architecture_name(name)


def test_check_architecture_name_experimental():
    with pytest.raises(
        ValueError, match="experimental architecture with the same name"
    ):
        check_architecture_name("soap_bpnn")


def test_check_architecture_name_deprecated():
    # Create once a deprecated architecture exist
    pass
