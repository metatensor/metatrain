import re

import pytest
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

from metatrain import soap_bpnn
from metatrain.utils import omegaconf
from metatrain.utils.omegaconf import (
    check_dataset_options,
    check_units,
    expand_dataset_config,
    expand_loss_config,
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
            assert targets_conf[target_key]["unit"] == ""
            assert targets_conf[target_key]["description"] == ""

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
    assert conf_expanded["systems"]["length_unit"] == ""

    targets_conf = conf_expanded["targets"]
    assert targets_conf["energy"]["quantity"] == "energy"
    assert targets_conf["energy"]["read_from"] == file_name
    assert targets_conf["energy"]["reader"] is None
    assert targets_conf["energy"]["key"] == "energy"
    assert targets_conf["energy"]["unit"] == ""
    assert targets_conf["energy"]["description"] == ""

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
        ValueError, match="Cannot perform training with respect to virial and stress"
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


def test_expand_loss_config_default():
    """
    When no custom loss is provided, architecture.training.loss
    should be created from the default template for each target in training_set.
    """
    conf = OmegaConf.create(
        {
            "training_set": {
                "targets": {
                    # no gradients requested
                    "energy": {"forces": False, "stress": False, "virial": False},
                    "dipole": {},  # non-energy target
                }
            },
            "architecture": {"training": {}},
        }
    )
    expanded = expand_loss_config(conf)
    loss = expanded["architecture"]["training"]["loss"]

    assert isinstance(loss, DictConfig)
    assert set(loss.keys()) == {"energy", "dipole"}

    # energy should have an empty gradients dict
    assert isinstance(loss["energy"]["gradients"], DictConfig)
    assert len(loss["energy"]["gradients"]) == 0

    # non-energy target gets the default loss template (resolved here)
    d = OmegaConf.to_container(loss["dipole"], resolve=True)
    assert d["type"] == "mse"
    assert d["weight"] == 1.0
    assert d["reduction"] == "mean"


def test_expand_loss_config_non_energy_only():
    """
    If the training_set contains only non-energy targets, no 'energy'
    block should appear in the final loss, only the non-energy ones.
    """
    conf = OmegaConf.create(
        {
            "training_set": {"targets": {"dipole": {}, "foo": {}}},
            "architecture": {"training": {}},
        }
    )
    expanded = expand_loss_config(conf)
    loss = expanded["architecture"]["training"]["loss"]

    assert "energy" not in loss
    assert set(loss.keys()) == {"dipole", "foo"}
    for t in ("dipole", "foo"):
        d = OmegaConf.to_container(loss[t], resolve=True)
        assert d["type"] == "mse"
        assert d["weight"] == 1.0
        assert d["reduction"] == "mean"


def test_expand_loss_config_single_string_applies_to_gradients():
    """
    When the loss is a single string, all targets and their gradients use that type.
    """
    conf = OmegaConf.create(
        {
            "training_set": {
                "targets": {
                    # energy with both gradients on
                    "energy": {"forces": {}, "stress": {}, "virial": False},
                    "dipole": {},
                }
            },
            "architecture": {"training": {"loss": "mae"}},
        }
    )
    expanded = expand_loss_config(conf)
    loss = expanded["architecture"]["training"]["loss"]

    assert set(loss.keys()) == {"energy", "dipole"}
    assert loss["energy"]["type"] == "mae"
    assert loss["dipole"]["type"] == "mae"

    # gradients inherit the same type
    pos = loss["energy"]["gradients"]["positions"]
    strain = loss["energy"]["gradients"]["strain"]
    assert pos["type"] == "mae"
    assert strain["type"] == "mae"


def test_expand_loss_config_per_target_string():
    """
    When the loss is given as a string per target, it expands to defaults with those
    types, and gradients (if any) keep their own defaults.
    """
    conf = OmegaConf.create(
        {
            "training_set": {
                "targets": {
                    "energy": {"forces": False, "stress": False, "virial": False},
                    "dipole": {},
                }
            },
            "architecture": {
                "training": {"loss": {"energy": "mse", "dipole": "huber"}}
            },
        }
    )
    expanded = expand_loss_config(conf)
    loss = expanded["architecture"]["training"]["loss"]
    assert set(loss.keys()) == {"energy", "dipole"}

    e = OmegaConf.to_container(loss["energy"], resolve=True)
    assert e["type"] == "mse"
    assert e["weight"] == 1.0
    assert e["reduction"] == "mean"

    d = OmegaConf.to_container(loss["dipole"], resolve=True)
    assert d["type"] == "huber"
    assert d["weight"] == 1.0
    assert d["reduction"] == "mean"
    assert "delta" in d and isinstance(d["delta"], (int, float))


def test_expand_loss_config_per_target_string_does_not_touch_gradients():
    """
    Per-target string type does not propagate to gradients; gradients keep defaults.
    """
    conf = OmegaConf.create(
        {
            "training_set": {
                "targets": {
                    "energy": {"forces": {}, "stress": False, "virial": False},
                }
            },
            "architecture": {"training": {"loss": {"energy": "mae"}}},
        }
    )
    expanded = expand_loss_config(conf)
    loss = expanded["architecture"]["training"]["loss"]
    e = loss["energy"]

    # scalar uses user type
    assert e["type"] == "mae"

    # gradient keeps default type (mse), not "mae"
    pos = OmegaConf.to_container(e["gradients"]["positions"], resolve=True)
    assert pos["type"] == "mse"
    assert pos["weight"] == 1.0
    assert pos["reduction"] == "mean"


def test_expand_loss_config_energy_forces_shorthand_string():
    """
    Energy target: `forces: <type>` shorthand maps to gradients.positions
    with that type, and does not affect the scalar loss.
    """
    conf = OmegaConf.create(
        {
            "training_set": {
                "targets": {
                    "mtt::energy-1": {
                        "quantity": "energy",
                        "forces": {},
                        "stress": False,
                        "virial": False,
                    }
                }
            },
            "architecture": {
                "training": {
                    "loss": {
                        "mtt::energy-1": {
                            "type": "mae",
                            "forces": "huber",
                        }
                    }
                }
            },
        }
    )
    expanded = expand_loss_config(conf)
    loss = expanded["architecture"]["training"]["loss"]
    e1 = loss["mtt::energy-1"]

    # scalar type from user
    assert e1["type"] == "mae"

    # forces shorthand -> gradients.positions
    pos = OmegaConf.to_container(e1["gradients"]["positions"], resolve=True)
    assert pos["type"] == "huber"
    # default weight/reduction come from CONF_LOSS
    assert pos["weight"] == 1.0
    assert pos["reduction"] == "mean"


def test_expand_loss_config_energy_forces_shorthand_dict():
    """
    Energy target: `forces: {...}` shorthand maps to gradients.positions
    with full dict, overriding defaults.
    """
    conf = OmegaConf.create(
        {
            "training_set": {
                "targets": {
                    "mtt::energy-1": {
                        "quantity": "energy",
                        "forces": {},
                        "stress": False,
                        "virial": False,
                    }
                }
            },
            "architecture": {
                "training": {
                    "loss": {
                        "mtt::energy-1": {
                            "type": "mse",
                            "forces": {
                                "type": "huber",
                                "weight": 2.0,
                            },
                        }
                    }
                }
            },
        }
    )
    expanded = expand_loss_config(conf)
    loss = expanded["architecture"]["training"]["loss"]
    e1 = loss["mtt::energy-1"]

    assert e1["type"] == "mse"

    pos = OmegaConf.to_container(e1["gradients"]["positions"], resolve=True)
    assert pos["type"] == "huber"
    assert pos["weight"] == 2.0
    assert pos["reduction"] == "mean"
    assert "delta" in pos and isinstance(pos["delta"], (int, float))


@pytest.mark.parametrize("grad_key", ["stress", "virial"])
def test_expand_loss_config_energy_stress_virial_shorthand(grad_key):
    """
    Energy target: `stress` or `virial` shorthand maps to gradients.strain.
    """
    conf = OmegaConf.create(
        {
            "training_set": {
                "targets": {
                    "mtt::etot": {
                        "quantity": "energy",
                        "forces": False,
                        "stress": {},
                        "virial": False,
                    }
                }
            },
            "architecture": {
                "training": {
                    "loss": {
                        "mtt::etot": {
                            "type": "mse",
                            grad_key: {
                                "type": "huber",
                                "weight": 0.3,
                            },
                        }
                    }
                }
            },
        }
    )
    expanded = expand_loss_config(conf)
    loss = expanded["architecture"]["training"]["loss"]
    et = loss["mtt::etot"]

    assert et["type"] == "mse"

    strain = OmegaConf.to_container(et["gradients"]["strain"], resolve=True)
    assert strain["type"] == "huber"
    assert strain["weight"] == 0.3
    assert strain["reduction"] == "mean"
    assert "delta" in strain and isinstance(strain["delta"], (int, float))


def test_expand_loss_config_gradients_override_shorthand():
    """
    Explicit gradients.<name> override energy shorthands (forces / stress).
    """
    conf = OmegaConf.create(
        {
            "training_set": {
                "targets": {
                    "energy": {
                        "forces": {},
                        "stress": {},
                        "virial": False,
                    }
                }
            },
            "architecture": {
                "training": {
                    "loss": {
                        "energy": {
                            "type": "mse",
                            "forces": {"type": "mae", "weight": 0.5},
                            "stress": "mae",
                            "gradients": {
                                "positions": {"type": "huber", "weight": 2.0},
                                "strain": "mse",
                            },
                        }
                    }
                }
            },
        }
    )
    expanded = expand_loss_config(conf)
    loss = expanded["architecture"]["training"]["loss"]
    e = loss["energy"]

    # scalar
    assert e["type"] == "mse"

    pos = OmegaConf.to_container(e["gradients"]["positions"], resolve=True)
    assert pos["type"] == "huber"
    assert pos["weight"] == 2.0  # overrides shorthand 0.5

    strain = OmegaConf.to_container(e["gradients"]["strain"], resolve=True)
    assert strain["type"] == "mse"  # overrides shorthand "mae"


def test_expand_loss_config_forces_on_non_energy_raises():
    """
    Using forces/stress/virial in loss for a non-energy target is an error.
    """
    conf = OmegaConf.create(
        {
            "training_set": {
                "targets": {
                    "dipole": {},  # not energy-like
                }
            },
            "architecture": {
                "training": {
                    "loss": {
                        "dipole": {
                            "type": "mse",
                            "forces": "mae",
                        }
                    }
                }
            },
        }
    )
    with pytest.raises(ValueError, match="only allowed for energy-like targets"):
        expand_loss_config(conf)


def test_expand_loss_config_stress_and_virial_together_raises():
    """
    Providing both stress and virial for the same energy target is forbidden.
    """
    conf = OmegaConf.create(
        {
            "training_set": {
                "targets": {
                    "energy": {
                        "forces": {},
                        "stress": {},
                        "virial": False,
                    }
                }
            },
            "architecture": {
                "training": {
                    "loss": {
                        "energy": {
                            "type": "mse",
                            "stress": "mae",
                            "virial": "mse",
                        }
                    }
                }
            },
        }
    )

    with pytest.raises(ValueError, match="Both 'stress' and 'virial' provided"):
        expand_loss_config(conf)


def test_expand_loss_config_huber_scalar_gets_delta(monkeypatch):
    """
    If a scalar uses huber without delta, it should get a default delta.
    """
    monkeypatch.setattr(
        "metatrain.utils.omegaconf.default_huber_loss_delta",
        lambda: 0.123,
    )

    conf = OmegaConf.create(
        {
            "training_set": {
                "targets": {
                    "energy": {"forces": False, "stress": False, "virial": False}
                }
            },
            "architecture": {
                "training": {"loss": {"energy": {"type": "huber"}}},
            },
        }
    )
    expanded = expand_loss_config(conf)
    e = OmegaConf.to_container(
        expanded["architecture"]["training"]["loss"]["energy"],
        resolve=True,
    )

    assert e["type"] == "huber"
    assert pytest.approx(e["delta"], rel=0, abs=1e-12) == 0.123


def test_expand_loss_config_huber_gradient_only_gets_delta(monkeypatch):
    """
    If only a gradient specifies huber without delta, it should get a default delta.
    """
    monkeypatch.setattr(
        "metatrain.utils.omegaconf.default_huber_loss_delta", lambda: 0.5
    )

    conf = OmegaConf.create(
        {
            "training_set": {
                "targets": {"energy": {"forces": {}, "stress": False, "virial": False}}
            },
            "architecture": {
                "training": {
                    "loss": {
                        "energy": {
                            "type": "mse",
                            "gradients": {"positions": {"type": "huber"}},
                        }
                    }
                }
            },
        }
    )
    expanded = expand_loss_config(conf)
    gpos = OmegaConf.to_container(
        expanded["architecture"]["training"]["loss"]["energy"]["gradients"][
            "positions"
        ],
        resolve=True,
    )
    assert gpos["type"] == "huber"
    assert gpos["delta"] == 0.5


def test_expand_loss_config_user_only_target_is_now_invalid():
    """
    A target defined only in loss (not in training_set) is no longer allowed:
    any loss key that does not correspond to an existing target must raise.
    """
    conf = OmegaConf.create(
        {
            "training_set": {"targets": {"dipole": {}}},  # only non-energy in dataset
            "architecture": {"training": {"loss": {"foo": "mae"}}},
        }
    )

    with pytest.raises(
        ValueError,
        match=re.escape(
            (
                "Invalid top-level loss entry 'foo'. "
                "Allowed keys are: ['dipole'] or a single string."
            )
        ),
    ):
        expand_loss_config(conf)


@pytest.mark.parametrize("bad_key", ["forces", "stress", "virial", "foo"])
def test_expand_loss_config_invalid_top_level_keys_raise(bad_key):
    """
    Top-level loss entries must either be:
      - a single string, e.g. loss: "mse"
      - a mapping whose keys are existing target names.

    Any other top-level key (including old shorthands like 'forces', 'stress',
    'virial', or arbitrary names not present in training_set.targets) must raise.
    """
    conf = OmegaConf.create(
        {
            "training_set": {
                "targets": {
                    "energy": {
                        "forces": {},
                        "stress": False,
                        "virial": False,
                    }
                }
            },
            "architecture": {
                "training": {
                    "loss": {
                        bad_key: "mae",
                    }
                }
            },
        }
    )

    with pytest.raises(
        ValueError, match=re.escape(f"Invalid top-level loss entry '{bad_key}'")
    ):
        expand_loss_config(conf)


def test_check_units():
    file_name = "foo.xyz"
    system_section = {"read_from": file_name, "length_unit": "angstrom"}

    target_section = {
        "quantity": "energy",
        "forces": file_name,
        "unit": "eV",
        "description": "my energy target",
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
