import ase
import ase.io
import torch

from metatensor.models.utils.data.readers.structures import read_structures_ase


def ase_system() -> ase.Atoms:
    symbols = ("H", "H")
    positions = [[0, 0, 0], [0, 0, 0.74]]
    info = {"true_energy": 42.0, "dipole_moment": 10.0}

    return ase.Atoms(symbols, positions=positions, info=info)


def test_read_ase(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    filename = "structures.xyz"

    structures = ase_system()
    ase.io.write(filename, structures)

    result = read_structures_ase(filename)

    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], torch.ScriptObject)

    torch.testing.assert_close(result[0].positions, torch.tensor(structures.positions))
    torch.testing.assert_close(
        result[0].species, torch.tensor([1, 1], dtype=torch.int32)
    )
