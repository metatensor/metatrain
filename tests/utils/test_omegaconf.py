import re

import pytest
import torch
from omegaconf import ListConfig, OmegaConf

from metatensor.models.experimental import soap_bpnn
from metatensor.models.utils import omegaconf
from metatensor.models.utils.omegaconf import (
    check_options_list,
    check_units,
    expand_dataset_config,
)


def test_file_format_resolver():
    conf = OmegaConf.create({"read_from": "foo.xyz", "file_format": "${file_format:}"})

    assert (conf["file_format"]) == ".xyz"


def test_random_seed_resolver():
    conf = OmegaConf.create({"seed": "${default_random_seed:}"})

    seed = conf["seed"]
    assert type(seed) is int
    assert seed > 0

    # assert that seed does not change if requested again
    assert seed == conf["seed"]


def test_default_device_resolver():
    conf = OmegaConf.create(
        {
            "device": "${default_device:}",
            "architecture": {"name": "experimental.soap_bpnn"},
        }
    )

    assert conf["device"] == "cpu"


def test_default_device_resolver_multi(monkeypatch):
    def pick_devices(architecture_devices):
        return [torch.device("cuda:0"), torch.device("cuda:1")]

    monkeypatch.setattr(omegaconf, "pick_devices", pick_devices)

    conf = OmegaConf.create(
        {
            "device": "${default_device:}",
            "architecture": {"name": "experimental.soap_bpnn"},
        }
    )

    assert conf["device"] == "multi-cuda"


@pytest.mark.parametrize(
    "dtype, precision",
    [(torch.float64, 64), (torch.double, 64), (torch.float32, 32), (torch.float16, 16)],
)
def test_default_precision_resolver(dtype, precision, monkeypatch):
    patched_capabilities = {"supported_dtypes": [dtype]}
    monkeypatch.setattr(soap_bpnn, "__capabilities__", patched_capabilities)

    conf = OmegaConf.create(
        {
            "base_precision": "${default_precision:}",
            "architecture": {"name": "experimental.soap_bpnn"},
        }
    )

    assert conf["base_precision"] == precision


def test_default_precision_resolver_unknown_dtype(monkeypatch):
    patched_capabilities = {"supported_dtypes": [torch.int64]}
    monkeypatch.setattr(soap_bpnn, "__capabilities__", patched_capabilities)

    conf = OmegaConf.create(
        {
            "base_precision": "${default_precision:}",
            "architecture": {"name": "experimental.soap_bpnn"},
        }
    )

    match = (
        r"architectures `default_dtype` \(torch.int64\) refers to an unknown torch "
        "dtype. This should not happen."
    )
    with pytest.raises(ValueError, match=match):
        conf["base_precision"]


@pytest.mark.parametrize("n_datasets", [1, 2])
def test_expand_dataset_config(n_datasets):
    """Test dataset expansion for a list of n_datasets times the same config"""
    file_name = "foo.xyz"
    file_format = ".xyz"

    system_section = {"read_from": file_name, "length_unit": "angstrom"}

    target_section = {
        "quantity": "energy",
        "forces": file_name,
        "virial": {"read_from": "my_grad.dat", "key": "foo"},
    }

    conf = {
        "systems": system_section,
        "targets": {"energy": target_section, "my_target": target_section},
    }

    conf = n_datasets * [conf]

    conf_expanded_list = expand_dataset_config(OmegaConf.create(conf))

    assert type(conf_expanded_list) is ListConfig
    assert len(conf_expanded_list) == n_datasets

    for conf_expanded in conf_expanded_list:
        assert conf_expanded["systems"]["read_from"] == file_name
        assert conf_expanded["systems"]["file_format"] == file_format
        assert conf_expanded["systems"]["length_unit"] == "angstrom"

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

        # If a virial is parsed as in the conf above the by default enabled section
        # "stress" should be disabled automatically
        assert targets_conf["energy"]["stress"] is False


def test_expand_dataset_config_not_energy():
    file_name = "foo.xyz"

    system_section = {"read_from": file_name, "unit": "angstrom"}

    target_section = {
        "quantity": "my_dipole_moment",
    }

    conf = {
        "systems": system_section,
        "targets": {"dipole_moment": target_section},
    }

    conf_expanded_list = expand_dataset_config(OmegaConf.create(conf))

    assert type(conf_expanded_list) is ListConfig
    assert len(conf_expanded_list) == 1
    conf_expanded = conf_expanded_list[0]

    assert conf_expanded["targets"]["dipole_moment"]["key"] == "dipole_moment"
    assert conf_expanded["targets"]["dipole_moment"]["quantity"] == "my_dipole_moment"
    assert conf_expanded["targets"]["dipole_moment"]["forces"] is False
    assert conf_expanded["targets"]["dipole_moment"]["stress"] is False
    assert conf_expanded["targets"]["dipole_moment"]["virial"] is False


def test_expand_dataset_config_min():
    file_name = "dataset.dat"
    file_format = ".dat"

    conf_expanded_list = expand_dataset_config(file_name)
    conf_expanded = conf_expanded_list[0]

    assert conf_expanded["systems"]["read_from"] == file_name
    assert conf_expanded["systems"]["file_format"] == file_format
    assert conf_expanded["systems"]["length_unit"] is None

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
        "systems": file_name,
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
        "systems": "foo.xyz",
        "targets": {
            "my_energy": {
                "forces": "data.txt",
                "virial": True,
                "stress": False,
            }
        },
    }

    conf_expanded_list = expand_dataset_config(OmegaConf.create(conf))
    conf_expanded = conf_expanded_list[0]

    assert conf_expanded["targets"]["my_energy"]["forces"]["read_from"] == "data.txt"
    assert conf_expanded["targets"]["my_energy"]["forces"]["file_format"] == ".txt"

    assert conf_expanded["targets"]["my_energy"]["stress"] is False
    conf_expanded["targets"]["my_energy"]["virial"]["read_from"]


def test_check_units():
    file_name = "foo.xyz"
    system_section = {"read_from": file_name, "length_unit": "angstrom"}

    target_section = {
        "quantity": "energy",
        "forces": file_name,
        "unit": "eV",
        "virial": {"read_from": "my_grad.dat", "key": "foo"},
    }

    mytarget_section = {
        "quantity": "love",
        "forces": file_name,
        "unit": "heart",
        "virial": {"read_from": "my_grad.dat", "key": "foo"},
    }

    conf = {
        "systems": system_section,
        "targets": {"energy": target_section, "my_target": mytarget_section},
    }

    system_section1 = {"read_from": file_name, "length_unit": "angstrom1"}

    target_section1 = {
        "quantity": "energy",
        "forces": file_name,
        "unit": "eV_",
        "virial": {"read_from": "my_grad.dat", "key": "foo"},
    }

    mytarget_section1 = {
        "quantity": "love",
        "forces": file_name,
        "unit": "heart_",
        "virial": {"read_from": "my_grad.dat", "key": "foo"},
    }

    conf1 = {
        "systems": system_section1,
        "targets": {"energy": target_section, "my_target": mytarget_section},
    }
    conf0 = {
        "systems": system_section,
        "targets": {"energy": target_section, "my_target0": mytarget_section},
    }
    conf2 = {
        "systems": system_section,
        "targets": {"energy": target_section1, "my_target": mytarget_section},
    }
    conf3 = {
        "systems": system_section,
        "targets": {"energy": target_section, "my_target": mytarget_section1},
    }

    train_options = expand_dataset_config(OmegaConf.create(conf))
    test_options = expand_dataset_config(OmegaConf.create(conf))

    test_options0 = expand_dataset_config(OmegaConf.create(conf0))

    test_options1 = expand_dataset_config(OmegaConf.create(conf1))
    test_options2 = expand_dataset_config(OmegaConf.create(conf2))
    test_options3 = expand_dataset_config(OmegaConf.create(conf3))

    check_units(actual_options=test_options, desired_options=train_options)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "`length_unit`s are inconsistent between one of the dataset options."
            " 'angstrom1' != 'angstrom'"
        ),
    ):
        check_units(actual_options=test_options1, desired_options=train_options)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Target 'my_target0' is not present in one of the given dataset options."
        ),
    ):
        check_units(actual_options=test_options0, desired_options=train_options)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Units of target 'energy' are inconsistent between one of the dataset "
            "options. 'eV_' != 'eV'."
        ),
    ):
        check_units(actual_options=test_options2, desired_options=train_options)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Units of target 'my_target' are inconsistent between one of the dataset "
            "options. 'heart_' != 'heart'."
        ),
    ):
        check_units(actual_options=test_options3, desired_options=train_options)


def test_missing_targets_section():
    conf = {"systems": "foo.xyz"}
    conf_expanded_list = expand_dataset_config(OmegaConf.create(conf))
    conf_expanded = conf_expanded_list[0]

    assert conf_expanded["systems"]["read_from"] == "foo.xyz"
    assert conf_expanded["systems"]["file_format"] == ".xyz"


def test_missing_strcutures_section():
    conf = {"targets": {"energies": "foo.xyz"}}
    conf_expanded_list = expand_dataset_config(OmegaConf.create(conf))
    conf_expanded = conf_expanded_list[0]

    assert conf_expanded["targets"]["energies"]["read_from"] == "foo.xyz"
    assert conf_expanded["targets"]["energies"]["file_format"] == ".xyz"


@pytest.fixture
def list_conf():
    file_name = "foo.xyz"

    system_section = {"read_from": file_name, "length_unit": "angstrom"}

    target_section = {
        "quantity": "energy",
        "unit": "eV",
        "forces": file_name,
        "virial": {"read_from": "my_grad.dat", "key": "foo"},
    }

    conf = {
        "systems": system_section,
        "targets": {"energy": target_section, "my_target": target_section},
    }

    return OmegaConf.create(3 * [conf])


def test_check_options_list_length_unit(list_conf):
    list_conf[1]["systems"]["length_unit"] = "foo"
    list_conf[2]["systems"]["length_unit"] = "bar"

    match = (
        "`length_unit`s are inconsistent between one of the dataset options. "
        "'foo' != 'angstrom'"
    )

    with pytest.raises(ValueError, match=match):
        check_options_list(list_conf)


def test_check_options_list_target_unit(list_conf):
    """Test three datasets where the unit of the 2nd and the 3rd is inconsistent."""
    list_conf[1]["targets"]["new_target"] = OmegaConf.create({"unit": "foo"})
    list_conf[2]["targets"]["new_target"] = OmegaConf.create({"unit": "bar"})

    match = (
        "Units of target section 'new_target' are inconsistent. Found "
        "'bar' and 'foo'"
    )

    with pytest.raises(ValueError, match=match):
        check_options_list(list_conf)
