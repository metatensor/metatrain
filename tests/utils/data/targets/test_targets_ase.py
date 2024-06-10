"""Here we check only the correct values. Correct shape and metadata will be checked
within `test_readers.py`"""

from typing import List

import ase.io
import pytest
import torch

from metatrain.utils.data.readers.targets import (
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

    filename = "systems.xyz"

    systems = ase_systems()
    ase.io.write(filename, systems)

    results = read_energy_ase(filename=filename, key="true_energy", dtype=torch.float16)

    for result, atoms in zip(results, systems):
        expected = torch.tensor([[atoms.info["true_energy"]]], dtype=torch.float16)
        torch.testing.assert_close(result.values, expected)


@pytest.mark.parametrize(
    "func, target_name",
    [
        (read_energy_ase, "energy"),
        (read_forces_ase, "forces"),
        (read_virial_ase, "virial"),
        (read_stress_ase, "stress"),
    ],
)
def test_ase_key_errors(func, target_name, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    filename = "systems.xyz"

    systems = ase_systems()
    ase.io.write(filename, systems)

    match = f"{target_name} key 'foo' was not found in system {filename!r} at index 0"

    with pytest.raises(KeyError, match=match):
        func(filename=filename, key="foo")


def test_read_forces_ase(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    filename = "systems.xyz"

    systems = ase_systems()
    ase.io.write(filename, systems)

    results = read_forces_ase(filename=filename, key="forces", dtype=torch.float16)

    for result, atoms in zip(results, systems):
        expected = -torch.tensor(atoms.get_array("forces"), dtype=torch.float16)
        expected = expected.reshape(-1, 3, 1)
        torch.testing.assert_close(result.values, expected)


def test_read_stress_ase(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    filename = "systems.xyz"

    systems = ase_systems()
    ase.io.write(filename, systems)

    results = read_stress_ase(filename=filename, key="stress-3x3", dtype=torch.float16)

    for result, atoms in zip(results, systems):
        expected = atoms.cell.volume * torch.tensor(
            atoms.info["stress-3x3"], dtype=torch.float16
        )
        expected = expected.reshape(-1, 3, 3, 1)
        torch.testing.assert_close(result.values, expected)


def test_no_cell_error(monkeypatch, tmp_path):
    """Test error raise if cell vectors are zero for reading stress."""
    monkeypatch.chdir(tmp_path)

    filename = "systems.xyz"

    systems = ase_system()
    systems.cell = [0.0, 0.0, 0.0]

    ase.io.write(filename, systems)

    with pytest.raises(ValueError, match="system 0 has zero cell vectors."):
        read_stress_ase(filename=filename, key="stress-3x3")


def test_read_virial_ase(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    filename = "systems.xyz"

    systems = ase_systems()
    ase.io.write(filename, systems)

    results = read_virial_ase(filename=filename, key="stress-3x3", dtype=torch.float16)

    for result, atoms in zip(results, systems):
        expected = -torch.tensor(atoms.info["stress-3x3"], dtype=torch.float16)
        expected = expected.reshape(-1, 3, 3, 1)
        torch.testing.assert_close(result.values, expected)


def test_read_virial_warn(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    filename = "systems.xyz"

    systems = ase_system()
    ase.io.write(filename, systems)

    with pytest.warns(match="Found 9-long numerical vector"):
        results = read_virial_ase(filename=filename, key="stress-9")

    expected = -torch.tensor(systems.info["stress-9"])
    expected = expected.reshape(-1, 3, 3, 1)
    torch.testing.assert_close(results[0].values, expected)


def test_read_virial_error(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    filename = "systems.xyz"

    systems = ase_system()
    systems.info["stress-9"].append(1)
    ase.io.write(filename, systems)

    with pytest.raises(ValueError, match="Stress/virial must be a 3 x 3 matrix"):
        read_virial_ase(filename=filename, key="stress-9")
