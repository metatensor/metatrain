"""Tests for the ASE readers. The functionality of the top-level functions
`read_systems`, `read_energy`, `read_generic` is already tested through
the reader tests in `test_readers.py`. Here we test the specific ASE readers
for energies, forces, stresses, and virials."""

from io import StringIO

import ase
import ase.io
import numpy as np
import pytest
import torch
from metatensor.torch import Labels
from test_targets_ase import ase_systems

from metatrain.utils.data.readers.ase import (
    _read_energy_ase,
    _read_forces_ase,
    _read_stress_ase,
    _read_virial_ase,
    read,
)


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


def test_read_restores_registered_extxyz_properties():
    header = (
        'dipole="1 2 3" free_energy=0.5 magmom=1.5 '
        'polarization="0 1 2" dielectric_tensor="1 0 0 0 1 0 0 0 1" '
        "Properties=species:S:1:pos:R:3:energies:R:1:stresses:R:6:"
        "charges:R:1:magmoms:R:1:born_effective_charges:R:9"
    )
    xyz = "\n".join(
        [
            "2",
            header,
            "H 0 0 0 0.1 1 2 3 4 5 6 0.2 1.0 1 0 0 0 1 0 0 0 1",
            "H 0 0 1 0.2 6 5 4 3 2 1 -0.2 -2.0 1 0 0 0 1 0 0 0 1",
        ]
    ) + "\n"

    atoms = read(StringIO(xyz), index=":", format="extxyz")[0]

    expected_arrays = {
        "energies": [0.1, 0.2],
        "stresses": [[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]],
        "charges": [0.2, -0.2],
        "magmoms": [1.0, -2.0],
        "born_effective_charges": [
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        ],
    }
    for key, expected in expected_arrays.items():
        np.testing.assert_allclose(atoms.arrays[key], expected)

    expected_info = {
        "free_energy": 0.5,
        "dipole": [1, 2, 3],
        "magmom": 1.5,
        "polarization": [0, 1, 2],
        "dielectric_tensor": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    }
    for key, expected in expected_info.items():
        np.testing.assert_allclose(atoms.info[key], expected)
