import shutil
import subprocess
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from metatensor.models.cli.train_model import expand_dataset_config


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


def test_expand_dataset_config():
    file_name = "foo.xyz"
    file_format = ".xyz"

    structure_section = {"read_from": file_name, "unit": "angstrom"}

    target_section = {
        "quantity": "energy",
        "forces": file_name,
        "virial": file_name,
        "bar": {"read_from": "my_grad.dat", "key": "foo"},
    }

    conf = {
        "structures": structure_section,
        "targets": {"energy": target_section, "energy2": target_section},
    }

    conf_expanded = expand_dataset_config(OmegaConf.create(conf))

    assert conf_expanded["structures"]["read_from"] == file_name
    assert conf_expanded["structures"]["file_format"] == file_format
    assert conf_expanded["structures"]["unit"] == "angstrom"

    targets_conf = conf_expanded["targets"]
    assert len(targets_conf) == 2

    assert targets_conf["energy"]["quantity"] == "energy"
    assert targets_conf["energy"]["read_from"] == file_name
    assert targets_conf["energy"]["file_format"] == file_format
    assert targets_conf["energy"]["file_format"] == file_format
    assert targets_conf["energy"]["key"] == "energy"
    assert targets_conf["energy"]["unit"] is None

    for gradient in ["forces", "virial"]:
        assert targets_conf["energy"][gradient]["read_from"] == file_name
        assert targets_conf["energy"][gradient]["file_format"] == file_format
        assert targets_conf["energy"][gradient]["key"] == gradient

    assert targets_conf["energy"]["bar"]["read_from"] == "my_grad.dat"
    assert targets_conf["energy"]["bar"]["key"] == "foo"

    # If a virial is parsed as in the conf above the by default enabled section "stress"
    # should be disabled automatically
    assert targets_conf["energy"]["stress"] is False

    assert targets_conf["energy2"]["key"] == "energy2"
    assert targets_conf["energy"]["quantity"] == "energy"


def test_expand_dataset_config_not_energy():
    file_name = "foo.xyz"

    structure_section = {"read_from": file_name, "unit": "angstrom"}

    target_section = {
        "quantity": "my_dipole_moment",
    }

    conf = {
        "structures": structure_section,
        "targets": {"dipole_moment": target_section},
    }

    conf_expanded = expand_dataset_config(OmegaConf.create(conf))

    assert conf_expanded["targets"]["dipole_moment"]["key"] == "dipole_moment"
    assert conf_expanded["targets"]["dipole_moment"]["quantity"] == "my_dipole_moment"
    assert conf_expanded["targets"]["dipole_moment"]["forces"] is False
    assert conf_expanded["targets"]["dipole_moment"]["stress"] is False
    assert conf_expanded["targets"]["dipole_moment"]["virial"] is False


def test_expand_dataset_config_min():
    file_name = "dataset.dat"
    file_format = ".dat"

    conf_expanded = expand_dataset_config(file_name)

    assert conf_expanded["structures"]["read_from"] == file_name
    assert conf_expanded["structures"]["file_format"] == file_format

    targets_conf = conf_expanded["targets"]
    assert targets_conf["energy"]["quantity"] == "energy"
    assert targets_conf["energy"]["read_from"] == file_name
    assert targets_conf["energy"]["file_format"] == file_format
    assert targets_conf["energy"]["file_format"] == file_format
    assert targets_conf["energy"]["key"] == "energy"
    assert targets_conf["energy"]["unit"] is None

    for gradient in ["forces", "stress"]:
        assert targets_conf["energy"][gradient]["read_from"] == file_name
        assert targets_conf["energy"][gradient]["file_format"] == file_format
        assert targets_conf["energy"][gradient]["key"] == gradient

    assert targets_conf["energy"]["virial"] is False


def test_expand_dataset_config_error():
    file_name = "foo.xyz"

    conf = {
        "structures": file_name,
        "targets": {
            "energy": {
                "virial": file_name,
                "stress": {"read_from": file_name, "key": "foo"},
            }
        },
    }

    with pytest.raises(
        ValueError, match="Cannot perform training with respect to virials and stress"
    ):
        expand_dataset_config(OmegaConf.create(conf))
