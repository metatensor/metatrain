"""Here we check only the correct values. Correct shape and metadata will be checked
within `test_readers.py`"""
from typing import List

import ase.io
import pytest
import torch

from metatensor.models.utils.data.readers.targets import (
    read_energy_ase,
    read_forces_ase,
    read_stress_ase,
    read_virial_ase,
)


def ase_system() -> ase.Atoms:
    symbols = ("H", "H")
    positions = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]
    info = {
        "true_energy": 42.0,
        "stress-3x3": [[0.0, 1.1, 2.1], [1.2, 2, 3], [13, 12, 12]],
        "stress-9": [0.1, 3, 6, 1, 2, 3, 4, 5, 53.3],
    }

    atoms = ase.Atoms(symbols, positions=positions, info=info, pbc=True, cell=[2, 2, 2])
    atoms.set_array("forces", positions, dtype=float)

    return atoms


def ase_systems() -> List[ase.Atoms]:
    return [ase_system(), ase_system()]


def test_read_energy_ase(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    filename = "structures.xyz"

    structures = ase_systems()
    ase.io.write(filename, structures)

    result = read_energy_ase(filename=filename, key="true_energy")

    l_expected = [a.info["true_energy"] for a in structures]
    expected = torch.tensor(l_expected).reshape(-1, 1)
    torch.testing.assert_close(result.values, expected)


def test_read_forces_ase(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    filename = "structures.xyz"

    structures = ase_systems()
    ase.io.write(filename, structures)

    result = read_forces_ase(filename=filename, key="forces")

    l_expected = [atoms.get_array("forces") for atoms in structures]
    expected = -torch.tensor(l_expected).reshape(-1, 3, 1)
    torch.testing.assert_close(result.values, expected)


def test_read_stress_ase(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    filename = "structures.xyz"

    structures = ase_systems()
    ase.io.write(filename, structures)

    result = read_stress_ase(filename=filename, key="stress-3x3")

    l_expected = [atoms.info["stress-3x3"] for atoms in structures]
    l_cell = [atoms.cell.volume for atoms in structures]
    expected = torch.tensor(l_expected)
    expected *= torch.tensor(l_cell).reshape(-1, 1, 1)
    expected = expected.reshape(-1, 3, 3, 1)
    torch.testing.assert_close(result.values, expected)


@pytest.mark.parametrize("reader", [read_stress_ase, read_virial_ase])
def test_no_cell_error(reader, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    filename = "structures.xyz"

    structures = ase_system()
    structures.cell = [0.0, 0.0, 0.0]

    ase.io.write(filename, structures)

    with pytest.raises(
        ValueError, match="Found at least one structure with zero cell vectors."
    ):
        read_stress_ase(filename=filename, key="stress-3x3")


def test_read_virial_ase(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    filename = "structures.xyz"

    structures = ase_systems()
    ase.io.write(filename, structures)

    result = read_virial_ase(filename=filename, key="stress-3x3")

    l_expected = [atoms.info["stress-3x3"] for atoms in structures]
    expected = -torch.tensor(l_expected)
    expected = expected.reshape(-1, 3, 3, 1)
    torch.testing.assert_close(result.values, expected)


def test_read_virial_warn(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    filename = "structures.xyz"

    structures = ase_system()
    ase.io.write(filename, structures)

    with pytest.warns(match="Found 9-long numerical vector"):
        result = read_virial_ase(filename=filename, key="stress-9")

    expected = -torch.tensor(structures.info["stress-9"]).reshape(3, 3)
    expected = expected.reshape(-1, 3, 3, 1)
    torch.testing.assert_close(result.values, expected)


def test_read_virial_error(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    filename = "structures.xyz"

    structures = ase_system()
    structures.info["stress-9"].append(1)
    ase.io.write(filename, structures)

    with pytest.raises(ValueError, match="stress/virial must be a 3 x 3 matrix"):
        read_virial_ase(filename=filename, key="stress-9")
