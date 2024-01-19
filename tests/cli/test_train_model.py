import shutil
import subprocess
from pathlib import Path

import pytest


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
