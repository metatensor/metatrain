import pytest
from omegaconf import OmegaConf

from metatensor.models.utils.omegaconf import expand_dataset_config


def test_file_format_resolver():
    conf = OmegaConf.create({"read_from": "foo.xyz", "file_format": "${file_format:}"})

    assert (conf["file_format"]) == ".xyz"


def test_expand_dataset_config():
    file_name = "foo.xyz"
    file_format = ".xyz"

    structure_section = {"read_from": file_name, "unit": "angstrom"}

    target_section = {
        "quantity": "energy",
        "forces": file_name,
        "virial": {"read_from": "my_grad.dat", "key": "foo"},
    }

    conf = {
        "structures": structure_section,
        "targets": {"energy": target_section, "my_target": target_section},
    }

    conf_expanded = expand_dataset_config(OmegaConf.create(conf))

    assert conf_expanded["structures"]["read_from"] == file_name
    assert conf_expanded["structures"]["file_format"] == file_format
    assert conf_expanded["structures"]["unit"] == "angstrom"

    targets_conf = conf_expanded["targets"]
    assert len(targets_conf) == 2

    for target_key in ["energy", "my_target"]:
        assert targets_conf[target_key]["quantity"] == "energy"
        assert targets_conf[target_key]["read_from"] == file_name
        assert targets_conf[target_key]["file_format"] == file_format
        assert targets_conf[target_key]["file_format"] == file_format
        assert targets_conf[target_key]["unit"] is None

        assert targets_conf[target_key]["forces"]["read_from"] == file_name
        assert targets_conf[target_key]["forces"]["file_format"] == file_format
        assert targets_conf[target_key]["forces"]["key"] == "forces"

        assert targets_conf[target_key]["virial"]["read_from"] == "my_grad.dat"
        assert targets_conf[target_key]["virial"]["file_format"] == ".dat"
        assert targets_conf[target_key]["virial"]["key"] == "foo"

        assert targets_conf[target_key]["stress"] is False

    # If a virial is parsed as in the conf above the by default enabled section "stress"
    # should be disabled automatically
    assert targets_conf["energy"]["stress"] is False


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


def test_expand_dataset_gradient():
    conf = {
        "structures": "foo.xyz",
        "targets": {
            "my_energy": {
                "forces": "data.txt",
                "virial": True,
                "stress": False,
            }
        },
    }

    conf_expanded = expand_dataset_config(OmegaConf.create(conf))

    assert conf_expanded["targets"]["my_energy"]["forces"]["read_from"] == "data.txt"
    assert conf_expanded["targets"]["my_energy"]["forces"]["file_format"] == ".txt"

    assert conf_expanded["targets"]["my_energy"]["stress"] is False
    conf_expanded["targets"]["my_energy"]["virial"]["read_from"]
