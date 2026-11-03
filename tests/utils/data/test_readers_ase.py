"""Tests for the ASE wrapper and specific energy/gradient readers.

The top-level readers are also tested through `test_readers.py`.
"""

import ase
import ase.io
import ase.io.extxyz
import numpy as np
import pytest
import torch
from ase.calculators.singlepoint import SinglePointCalculator
from metatensor.torch import Labels
from test_targets_ase import ase_systems

from metatrain.utils.data.readers.ase import (
    _read_energy_ase,
    _read_forces_ase,
    _read_stress_ase,
    _read_virial_ase,
    read,
)


@pytest.mark.parametrize("ase_version", ["3.22", "3.23-3.24", "3.25+"])
@pytest.mark.parametrize(
    "read_args",
    [(), (0,), (":",)],
    ids=["default-frame", "indexed-frame", "all-frames"],
)
def test_restore_calculator_results(monkeypatch, ase_version, read_args):
    # ASE 3.22 uses hardcoded lists and retains results in info/arrays:
    # https://gitlab.com/ase/ase/-/blob/3.22.1/ase/io/extxyz.py#L50-53
    per_atom = ["forces", "stresses", "charges", "magmoms", "energies"]
    per_config = ["energy", "stress", "dipole", "magmom", "free_energy"]
    if ase_version != "3.22":
        # ASE 3.23 moves results out of info/arrays and derives the lists from
        # all_outputs, including the newly supported tensor properties:
        # https://gitlab.com/ase/ase/-/blob/3.23.0/ase/io/extxyz.py#L47-55
        # https://gitlab.com/ase/ase/-/blob/3.23.0/ase/io/extxyz.py#L501-526
        per_atom += ["born_effective_charges"]
        per_config += ["polarization", "dielectric_tensor"]
    if ase_version == "3.23-3.24":
        per_config += [
            "natoms",
            "nbands",
            "nkpts",
            "nspins",
            "fermi_level",
            "kpoint_weights",
            "ibz_kpoints",
            "eigenvalues",
            "occupations",
        ]
    # ASE 3.25 filters the lists to properties accepted by SinglePointCalculator:
    # https://gitlab.com/ase/ase/-/blob/3.25.0/ase/io/extxyz.py#L51-62
    monkeypatch.setattr(ase.io.extxyz, "per_atom_properties", per_atom)
    monkeypatch.setattr(ase.io.extxyz, "per_config_properties", per_config)

    atom_data = {
        "forces": np.arange(6).reshape(2, 3),
        "stresses": np.arange(12).reshape(2, 6),
        "charges": np.array([0.2, -0.2]),
        "magmoms": np.array([1.0, -2.0]),
        "energies": np.array([-1.0, -2.0]),
        "born_effective_charges": np.arange(18).reshape(2, 9),
    }
    config_data = {
        "energy": -3.0,
        "stress": np.arange(6),
        "dipole": np.array([1.0, 2.0, 3.0]),
        "magmom": -1.0,
        "free_energy": -3.1,
        "polarization": np.array([4.0, 5.0, 6.0]),
        "dielectric_tensor": np.arange(9),
    }
    atoms = ase.Atoms("H2")
    results = {**atom_data, **config_data}
    if ase_version == "3.22":
        # Charges become initial_charges, not calculator results, in ASE 3.22:
        # https://gitlab.com/ase/ase/-/blob/3.22.1/ase/io/extxyz.py#L477-490
        atoms.arrays.update(atom_data)
        atoms.arrays["initial_charges"] = atoms.arrays.pop("charges")
        atoms.info.update(config_data)
        atoms.info["stress"] = np.zeros((3, 3))
        for key in [
            "charges",
            "born_effective_charges",
            "polarization",
            "dielectric_tensor",
        ]:
            results.pop(key)
    atoms.calc = SinglePointCalculator(atoms, **results)
    # Mixed datasets may also contain frames without any calculator results.
    empty = ase.Atoms("H")
    single_frame = read_args != (":",)
    frames = atoms if single_frame else [atoms, empty]
    monkeypatch.setattr(ase.io, "read", lambda *args, **kwargs: frames)

    result = read("unused.extxyz", *read_args)
    if single_frame:
        assert isinstance(result, ase.Atoms)
        restored = result
    else:
        assert isinstance(result, list)
        restored, without_results = result
        assert without_results.info == {}
        assert set(without_results.arrays) == {"numbers", "positions"}

    for key, expected in atom_data.items():
        if key == "charges" and ase_version == "3.22":
            assert "charges" not in restored.arrays
            np.testing.assert_array_equal(restored.arrays["initial_charges"], expected)
        else:
            np.testing.assert_array_equal(restored.arrays[key], expected)
        assert key not in restored.info
    for key, expected in config_data.items():
        if key == "stress":
            assert restored.info[key].shape == (3, 3)
            assert restored.calc.results[key].shape == (6,)
        else:
            np.testing.assert_array_equal(restored.info[key], expected)
        assert key not in restored.arrays


@pytest.mark.parametrize("key", ["true_energy", "energy"])
def test_read_energies(monkeypatch, tmp_path, key):
    monkeypatch.chdir(tmp_path)

    filename = "systems.xyz"
    systems = ase_systems()
    ase.io.write(filename, systems)

    results = _read_energy_ase(filename, key=key)

    assert type(results) is list
    assert len(results) == len(systems)
    for i_system, result in enumerate(results):
        assert result.values.dtype is torch.float64
        assert result.samples.names == ["system"]
        assert result.samples.values == torch.tensor([[i_system]])
        assert result.properties == Labels("energy", torch.tensor([[0]]))


@pytest.mark.parametrize("key", ["true_forces", "forces"])
def test_read_forces(monkeypatch, tmp_path, key):
    monkeypatch.chdir(tmp_path)

    filename = "systems.xyz"
    systems = ase_systems()
    ase.io.write(filename, systems)

    results = _read_forces_ase(filename, key=key)

    assert type(results) is list
    assert len(results) == len(systems)
    for i_system, result in enumerate(results):
        assert result.values.dtype is torch.float64
        assert result.samples.names == ["sample", "system", "atom"]
        assert torch.all(result.samples["sample"] == torch.tensor(0))
        assert torch.all(result.samples["system"] == torch.tensor(i_system))
        assert result.components == [Labels(["xyz"], torch.arange(3).reshape(-1, 1))]
        assert result.properties == Labels("energy", torch.tensor([[0]]))


@pytest.mark.parametrize("key", ["stress", "stress-3x3"])
@pytest.mark.parametrize("reader_func", [_read_stress_ase, _read_virial_ase])
def test_read_stress_virial(reader_func, monkeypatch, tmp_path, key):
    monkeypatch.chdir(tmp_path)

    filename = "systems.xyz"
    systems = ase_systems()
    ase.io.write(filename, systems)

    results = reader_func(filename, key=key)

    assert type(results) is list
    assert len(results) == len(systems)
    components = [
        Labels(["xyz_1"], torch.arange(3).reshape(-1, 1)),
        Labels(["xyz_2"], torch.arange(3).reshape(-1, 1)),
    ]
    for result in results:
        assert result.values.dtype is torch.float64
        assert result.samples.names == ["sample"]
        assert result.samples.values == torch.tensor([[0]])
        assert result.components == components
        assert result.properties == Labels("energy", torch.tensor([[0]]))
