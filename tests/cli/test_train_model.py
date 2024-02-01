import shutil
import subprocess
from pathlib import Path

import ase.io
import pytest
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


# TODO: test split of train/test/validation using floats and combinations of these.
