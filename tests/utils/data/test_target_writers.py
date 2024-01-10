from typing import List

import ase.io
import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.atomistic import System
from rascaline.torch import systems_to_torch

from metatensor.models.utils.data.writers import write_predictions, write_xyz


def structures_predictions(cell: torch.tensor = None) -> List[System]:
    if cell is None:
        cell = torch.zeros(3, 3)

    structures = systems_to_torch(
        2
        * [
            System(
                species=torch.tensor([1, 1]),
                positions=torch.tensor([[0, 0, 0], [0, 0, 0.74]]),
                cell=cell,
            ),
        ]
    )

    # Create a mock TensorMap for predictions
    n_structures = len(structures)
    values = torch.tensor([[1.0], [2.0]])
    block = TensorBlock(
        values=values.reshape(-1, 1),
        samples=Labels(["structure"], torch.arange(n_structures).reshape(-1, 1)),
        components=[],
        properties=Labels(["energy"], torch.tensor([(0,)])),
    )

    predictions = {"energy": TensorMap(Labels.single(), [block])}

    return structures, predictions


def test_write_xyz(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    structures, predictions = structures_predictions()

    filename = "test_output.xyz"

    write_xyz(filename, predictions, structures)

    # Read the file and verify its contents
    frames = ase.io.read(filename, index=":")
    assert len(frames) == len(structures)
    for i, atoms in enumerate(frames):
        assert atoms.info["energy"] == float(predictions["energy"].block().values[i, 0])
        assert all(atoms.pbc == 3 * [False])


def test_write_xyz_cell(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    cell = torch.ones(3, 3)
    structures, predictions = structures_predictions(cell=cell)

    filename = "test_output.xyz"

    write_xyz(filename, predictions, structures)

    # Read the file and verify its contents
    frames = ase.io.read(filename, index=":")
    for atoms in frames:
        torch.testing.assert_close(torch.tensor(atoms.cell[:]), cell)
        assert all(atoms.pbc == 3 * [True])


@pytest.mark.parametrize("fileformat", (None, ".xyz"))
def test_write_predictions(fileformat, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    structures, predictions = structures_predictions()

    filename = "test_output.xyz"

    write_predictions(filename, predictions, structures, fileformat=fileformat)

    frames = ase.io.read(filename, index=":")
    assert len(frames) == len(structures)
    for i, frame in enumerate(frames):
        assert frame.info["energy"] == float(predictions["energy"].block().values[i, 0])


def test_write_predictions_unknown_fileformat():
    with pytest.raises(ValueError, match="fileformat '.bar' is not supported"):
        write_predictions("foo.bar", predictions=None, structures=None)
