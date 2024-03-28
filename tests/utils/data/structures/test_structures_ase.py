import ase
import ase.io
import torch

from metatensor.models.utils.data.readers.systems import read_systems_ase


def ase_system() -> ase.Atoms:
    symbols = ("H", "H")
    positions = [[0, 0, 0], [0, 0, 0.74]]
    info = {"true_energy": 42.0, "dipole_moment": 10.0}

    return ase.Atoms(symbols, positions=positions, info=info)


def test_read_ase(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    filename = "systems.xyz"

    systems = ase_system()
    ase.io.write(filename, systems)

    result = read_systems_ase(filename)

    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], torch.ScriptObject)

    positions_actual = result[0].positions
    positions_expected = torch.tensor(systems.positions, dtype=positions_actual.dtype)
    torch.testing.assert_close(positions_actual, positions_expected)

    types_expected = result[0].types
    types_actual = torch.tensor([1, 1], dtype=types_expected.dtype)
    torch.testing.assert_close(types_expected, types_actual)
