"""Tests for the ASE readers. The functionality of the top-level functions
`read_systems`, `read_energy`, `read_generic` is already tested through
the reader tests in `test_readers.py`. Here we test the specific ASE readers
for energies, forces, stresses, and virials."""

import ase
import ase.io
import pytest
import torch
from metatensor.torch import Labels
from test_targets_ase import ase_systems

from metatrain.utils.data.readers.ase import (
    read_energy_ase,
    read_forces_ase,
    read_stress_ase,
    read_virial_ase,
)


def test_read_energies(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    filename = "systems.xyz"
    systems = ase_systems()
    ase.io.write(filename, systems)

    results = read_energy_ase(filename, key="true_energy")

    assert type(results) is list
    assert len(results) == len(systems)
    for i_system, result in enumerate(results):
        assert result.values.dtype is torch.float64
        assert result.samples.names == ["system"]
        assert result.samples.values == torch.tensor([[i_system]])
        assert result.properties == Labels("energy", torch.tensor([[0]]))


def test_read_forces(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    filename = "systems.xyz"
    systems = ase_systems()
    ase.io.write(filename, systems)

    results = read_forces_ase(filename, key="forces")

    assert type(results) is list
    assert len(results) == len(systems)
    for i_system, result in enumerate(results):
        assert result.values.dtype is torch.float64
        assert result.samples.names == ["sample", "system", "atom"]
        assert torch.all(result.samples["sample"] == torch.tensor(0))
        assert torch.all(result.samples["system"] == torch.tensor(i_system))
        assert result.components == [Labels(["xyz"], torch.arange(3).reshape(-1, 1))]
        assert result.properties == Labels("energy", torch.tensor([[0]]))


@pytest.mark.parametrize("reader_func", [read_stress_ase, read_virial_ase])
def test_read_stress_virial(reader_func, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    filename = "systems.xyz"
    systems = ase_systems()
    ase.io.write(filename, systems)

    results = reader_func(filename, key="stress-3x3")

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
