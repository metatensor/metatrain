import glob
import shutil
import subprocess
from pathlib import Path

import ase.io
import metatensor.torch  # noqa
import pytest
import torch
from omegaconf import OmegaConf


RESOURCES_PATH = Path(__file__).parent.resolve() / ".." / "resources"


@pytest.mark.parametrize("output", [None, "mymodel.pt"])
def test_train(monkeypatch, tmp_path, output):
    """Test that training via the training cli runs without an error raise."""
    monkeypatch.chdir(tmp_path)
    shutil.copy(RESOURCES_PATH / "qm9_reduced_100.xyz", "qm9_reduced_100.xyz")
    shutil.copy(RESOURCES_PATH / "options.yaml", "options.yaml")

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


@pytest.mark.parametrize("test_set_file", (True, False))
@pytest.mark.parametrize("validation_set_file", (True, False))
@pytest.mark.parametrize("output", [None, "mymodel.pt"])
def test_train_explicit_validation_test(
    monkeypatch, tmp_path, test_set_file, validation_set_file, output
):
    """Test that training via the training cli runs without an error raise
    also when the validation and test sets are provided explicitly."""
    monkeypatch.chdir(tmp_path)

    structures = ase.io.read(RESOURCES_PATH / "qm9_reduced_100.xyz", ":")
    options = OmegaConf.load(RESOURCES_PATH / "options.yaml")

    ase.io.write("qm9_reduced_100.xyz", structures[:50])

    if test_set_file:
        ase.io.write("test.xyz", structures[50:80])
        options["validation_set"] = options["training_set"].copy()
        options["validation_set"]["structures"]["read_from"] = "test.xyz"

    if validation_set_file:
        ase.io.write("validation.xyz", structures[80:])
        options["test_set"] = options["training_set"].copy()
        options["test_set"]["structures"]["read_from"] = "validation.xyz"

    OmegaConf.save(config=options, f="options.yaml")
    command = ["metatensor-models", "train", "options.yaml"]

    if output is not None:
        command += ["-o", output]
    else:
        output = "model.pt"

    subprocess.check_call(command)
    assert Path(output).is_file()


def test_continue(monkeypatch, tmp_path):
    """Test that continuing training from a checkpoint runs without an error raise."""
    monkeypatch.chdir(tmp_path)
    shutil.copy(RESOURCES_PATH / "qm9_reduced_100.xyz", "qm9_reduced_100.xyz")
    shutil.copy(RESOURCES_PATH / "bpnn-model.pt", "bpnn-model.pt")
    shutil.copy(RESOURCES_PATH / "options.yaml", "options.yaml")

    command = ["metatensor-models", "train", "options.yaml", "-c bpnn-model.pt"]
    subprocess.check_call(command)


def test_continue_different_dataset(monkeypatch, tmp_path):
    """Test that continuing training from a checkpoint runs without an error raise
    with a different dataset than the original."""
    monkeypatch.chdir(tmp_path)
    shutil.copy(RESOURCES_PATH / "ethanol_reduced_100.xyz", "ethanol_reduced_100.xyz")
    shutil.copy(
        RESOURCES_PATH / "bpnn-model.pt",
        "bpnn-model.pt",
    )

    options = OmegaConf.load(RESOURCES_PATH / "options.yaml")
    options["training_set"]["structures"]["read_from"] = "ethanol_reduced_100.xyz"
    options["training_set"]["targets"]["energy"]["key"] = "energy"
    print(options)
    OmegaConf.save(config=options, f="options.yaml")

    command = [
        "metatensor-models",
        "train",
        "options.yaml",
        "-c bpnn-model.pt",
    ]
    subprocess.check_call(command)


def test_yml_error():
    """Test error raise of the option file is not a .yaml file."""
    try:
        subprocess.check_output(
            ["metatensor-models", "train", "options.yml"], stderr=subprocess.STDOUT
        )
    except subprocess.CalledProcessError as captured:
        assert "Options file 'options.yml' must be a `.yaml` file." in str(
            captured.output
        )


def test_hydra_arguments():
    """Test if hydra arguments work."""
    option_path = str(RESOURCES_PATH / "options.yaml")
    out = subprocess.check_output(
        ["metatensor-models", "train", option_path, "--hydra=--help"]
    )
    # Check that num_epochs is override is succesful
    assert "num_epochs: 1" in str(out)


def test_no_architecture_name(monkeypatch, tmp_path):
    """Test error raise if architecture.name is not set."""
    monkeypatch.chdir(tmp_path)

    options = OmegaConf.load(RESOURCES_PATH / "options.yaml")
    options["architecture"].pop("name")
    OmegaConf.save(config=options, f="options.yaml")

    try:
        subprocess.check_output(
            ["metatensor-models", "train", "options.yaml"], stderr=subprocess.STDOUT
        )
    except subprocess.CalledProcessError as captured:
        assert "Architecture name is not defined!" in str(captured.output)


@pytest.mark.parametrize("seed", [1234, -1, -123])
@pytest.mark.parametrize("architecture_name", ["soap_bpnn"])
def test_model_consistency_with_seed(monkeypatch, tmp_path, architecture_name, seed):
    """Checks final model consistency with a fixed seed."""
    monkeypatch.chdir(tmp_path)
    shutil.copy(RESOURCES_PATH / "qm9_reduced_100.xyz", "qm9_reduced_100.xyz")

    options = OmegaConf.load(RESOURCES_PATH / "options.yaml")
    options["architecture"]["name"] = architecture_name
    options["seed"] = seed
    OmegaConf.save(config=options, f="options.yaml")

    if seed < -1:
        try:
            subprocess.check_output(
                ["metatensor-models", "train", "options.yaml", "-o", "model1.pt"],
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as captured:
            assert "should be a positive number or -1." in str(captured.output)
    else:
        subprocess.check_call(
            ["metatensor-models", "train", "options.yaml", "-o", "model1.pt"]
        )
        subprocess.check_call(
            ["metatensor-models", "train", "options.yaml", "-o", "model2.pt"]
        )

        m1 = torch.load("model1.pt")
        m2 = torch.load("model2.pt")

        for index, i in enumerate(m1["model_state_dict"]):
            tensor1 = m1["model_state_dict"][i]
            tensor2 = m2["model_state_dict"][i]

            # The first tensor only depend on the chemical compositions (not on the
            # seed) and should alwyas be the same.
            if seed > -1 or index == 0:
                assert torch.allclose(tensor1, tensor2)
            else:
                assert not torch.allclose(tensor1, tensor2)


def test_error_base_precision(monkeypatch, tmp_path):
    """Test unsopperted base_precision"""
    monkeypatch.chdir(tmp_path)

    options = OmegaConf.load(RESOURCES_PATH / "options.yaml")
    options["base_precision"] = "123"
    OmegaConf.save(config=options, f="options.yaml")

    try:
        subprocess.check_output(
            ["metatensor-models", "train", "options.yaml"], stderr=subprocess.STDOUT
        )
    except subprocess.CalledProcessError as captured:
        assert "Only 64, 32 or 16 are possible values for " in str(captured.output)
