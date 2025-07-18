import re

import pytest
import torch
from omegaconf import ListConfig, OmegaConf

from metatrain import soap_bpnn
from metatrain.utils import omegaconf
from metatrain.utils.omegaconf import (
    check_dataset_options,
    check_units,
    expand_dataset_config,
)


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
            "architecture": {"name": "soap_bpnn"},
        }
    )

    assert conf["device"] == "cuda" if torch.cuda.is_available() else "cpu"


def test_default_device_resolver_multi(monkeypatch):
    def pick_devices(architecture_devices):
        return [torch.device("cuda:0"), torch.device("cuda:1")]

    monkeypatch.setattr(omegaconf, "pick_devices", pick_devices)

    conf = OmegaConf.create(
        {
            "device": "${default_device:}",
            "architecture": {"name": "soap_bpnn"},
        }
    )

    assert conf["device"] == "multi-cuda"


@pytest.mark.parametrize(
    "dtype, precision",
    [(torch.float64, 64), (torch.double, 64), (torch.float32, 32), (torch.float16, 16)],
)
def test_default_precision_resolver(dtype, precision, monkeypatch):
    monkeypatch.setattr(soap_bpnn.__model__, "__supported_dtypes__", [dtype])

    conf = OmegaConf.create(
        {
            "base_precision": "${default_precision:}",
            "architecture": {"name": "soap_bpnn"},
        }
    )

    assert conf["base_precision"] == precision


def test_default_precision_resolver_unknown_dtype(monkeypatch):
    monkeypatch.setattr(soap_bpnn.__model__, "__supported_dtypes__", [torch.int64])

    conf = OmegaConf.create(
        {
            "base_precision": "${default_precision:}",
            "architecture": {"name": "soap_bpnn"},
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

    system_section = {"read_from": file_name, "length_unit": "angstrom"}

    target_section = {
        "quantity": "energy",
        "forces": file_name,
        "virial": {"read_from": "my_grad.extxyz", "key": "foo"},
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
        assert conf_expanded["systems"]["reader"] is None
        assert conf_expanded["systems"]["length_unit"] == "angstrom"

        targets_conf = conf_expanded["targets"]
        assert len(targets_conf) == 2

        for target_key in ["energy", "my_target"]:
            assert targets_conf[target_key]["quantity"] == "energy"
            assert targets_conf[target_key]["read_from"] == file_name
            assert targets_conf[target_key]["reader"] is None
            assert targets_conf[target_key]["unit"] is None

            assert targets_conf[target_key]["forces"]["read_from"] == file_name
            assert targets_conf[target_key]["forces"]["reader"] is None
            assert targets_conf[target_key]["forces"]["key"] == "forces"

            assert targets_conf[target_key]["virial"]["read_from"] == "my_grad.extxyz"
            assert targets_conf[target_key]["virial"]["reader"] is None
            assert targets_conf[target_key]["virial"]["key"] == "foo"

            assert targets_conf[target_key]["stress"] is False

        # If a virial is parsed as in the conf above the by default enabled section
        # "stress" should be disabled automatically
        assert targets_conf["energy"]["stress"] is False


def test_expand_dataset_config_not_energy():
    file_name = "foo.xyz"

    system_section = {"read_from": file_name, "length_unit": "angstrom"}

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

    conf_expanded_list = expand_dataset_config(file_name)
    conf_expanded = conf_expanded_list[0]

    assert conf_expanded["systems"]["read_from"] == file_name
    assert conf_expanded["systems"]["reader"] is None
    assert conf_expanded["systems"]["length_unit"] is None

    targets_conf = conf_expanded["targets"]
    assert targets_conf["energy"]["quantity"] == "energy"
    assert targets_conf["energy"]["read_from"] == file_name
    assert targets_conf["energy"]["reader"] is None
    assert targets_conf["energy"]["key"] == "energy"
    assert targets_conf["energy"]["unit"] is None

    for gradient in ["forces", "stress"]:
        assert targets_conf["energy"][gradient]["read_from"] == file_name
        assert targets_conf["energy"][gradient]["reader"] is None
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
                "forces": "data.xyz",
                "virial": True,
                "stress": False,
            }
        },
    }

    conf_expanded_list = expand_dataset_config(OmegaConf.create(conf))
    conf_expanded = conf_expanded_list[0]

    assert conf_expanded["targets"]["my_energy"]["forces"]["read_from"] == "data.xyz"
    assert conf_expanded["targets"]["my_energy"]["forces"]["reader"] is None

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


def test_error_target_and_mtt_target():
    file_name = "foo.xyz"
    system_section = {"read_from": file_name, "length_unit": "angstrom"}

    energy_section = {
        "quantity": "energy",
        "forces": file_name,
        "unit": "eV",
        "virial": {"read_from": "my_grad.dat", "key": "foo"},
    }

    conf = {
        "systems": system_section,
        "targets": {"energy": energy_section, "mtt::energy": energy_section},
    }

    with pytest.raises(
        ValueError,
        match=(
            "Two targets with the names `energy` and `mtt::energy` "
            "are not allowed to be present at the same time."
        ),
    ):
        expand_dataset_config(OmegaConf.create(conf))


def test_missing_targets_section():
    conf = {"systems": "foo.xyz"}
    conf_expanded_list = expand_dataset_config(OmegaConf.create(conf))
    conf_expanded = conf_expanded_list[0]

    assert conf_expanded["systems"]["read_from"] == "foo.xyz"
    assert conf_expanded["systems"]["reader"] is None


def test_missing_strcutures_section():
    conf = {"targets": {"energies": "foo.xyz"}}
    conf_expanded_list = expand_dataset_config(OmegaConf.create(conf))
    conf_expanded = conf_expanded_list[0]

    assert conf_expanded["targets"]["energies"]["read_from"] == "foo.xyz"
    assert conf_expanded["targets"]["energies"]["reader"] is None


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

    extra_data_section = {
        "quantity": "",
        "unit": "eV",
    }

    conf = {
        "systems": system_section,
        "targets": {"energy": target_section, "my_target": target_section},
        "extra_data": {"extra-data": extra_data_section},
    }

    return OmegaConf.create(3 * [conf])


@pytest.mark.parametrize("func", [expand_dataset_config, check_dataset_options])
def test_check_dataset_options_length_unit(func, list_conf):
    list_conf[1]["systems"]["length_unit"] = "foo"
    list_conf[2]["systems"]["length_unit"] = "bar"

    match = (
        "`length_unit`s are inconsistent between one of the dataset options. "
        "'foo' != 'angstrom'"
    )

    with pytest.raises(ValueError, match=match):
        func(list_conf)


def test_check_dataset_options_target_unit(list_conf):
    """Test three datasets where the unit of the 2nd and the 3rd is inconsistent."""
    list_conf[1]["targets"]["new_target"] = OmegaConf.create({"unit": "foo"})
    list_conf[2]["targets"]["new_target"] = OmegaConf.create({"unit": "bar"})

    match = (
        "Units of target section 'new_target' are inconsistent. Found 'bar' and 'foo'"
    )

    with pytest.raises(ValueError, match=match):
        check_dataset_options(list_conf)


def test_check_dataset_options_extra_data_unit(list_conf):
    """Test three datasets where the unit of the 2nd and the 3rd is inconsistent."""
    list_conf[1]["extra_data"]["new_data"] = OmegaConf.create({"unit": "foo"})
    list_conf[2]["extra_data"]["new_data"] = OmegaConf.create({"unit": "bar"})

    match = (
        "Units of extra_data section 'new_data' are inconsistent. "
        "Found 'bar' and 'foo'!"
    )

    with pytest.raises(ValueError, match=match):
        check_dataset_options(list_conf)


def generate_reference_config():
    return OmegaConf.create(
        {
            "componentA": {
                "setting1": "value1",
                "setting2": 1234,
                "sub_component": {"sub_setting1": "value2"},
            },
            "componentB": {"option1": True, "option2": "enabled"},
            "componentC": [{"item1": "value1"}, {"item2": "value2"}],
            "componentD": ["value1", "value2"],
        }
    )
