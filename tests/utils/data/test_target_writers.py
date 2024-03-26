from typing import List

import ase.io
import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.atomistic import System

from metatensor.models.utils.data.writers import write_predictions, write_xyz


def systems_predictions(cell: torch.tensor = None) -> List[System]:
    if cell is None:
        cell = torch.zeros((3, 3))

    systems = 2 * [
        System(
            types=torch.tensor([1, 1]),
            positions=torch.tensor([[0, 0, 0], [0, 0, 0.74]]),
            cell=cell,
        ),
    ]

    # Create a mock TensorMap for predictions
    n_systems = len(systems)
    values = torch.tensor([[1.0], [2.0]])
    block = TensorBlock(
        values=values.reshape(-1, 1),
        samples=Labels(["system"], torch.arange(n_systems).reshape(-1, 1)),
        components=[],
        properties=Labels(["energy"], torch.tensor([(0,)])),
    )

    predictions = {"energy": TensorMap(Labels.single(), [block])}

    return systems, predictions


def test_write_xyz(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    systems, predictions = systems_predictions()

    filename = "test_output.xyz"

    write_xyz(filename, predictions, systems)

    # Read the file and verify its contents
    frames = ase.io.read(filename, index=":")
    assert len(frames) == len(systems)
    for i, atoms in enumerate(frames):
        assert atoms.info["energy"] == float(predictions["energy"].block().values[i, 0])
        assert all(atoms.pbc == 3 * [False])


def test_write_xyz_cell(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    cell_expected = torch.ones(3, 3)
    systems, predictions = systems_predictions(cell=cell_expected)

    filename = "test_output.xyz"

    write_xyz(filename, predictions, systems)

    # Read the file and verify its contents
    frames = ase.io.read(filename, index=":")
    for atoms in frames:
        cell_actual = torch.tensor(atoms.cell, dtype=cell_expected.dtype)
        torch.testing.assert_close(cell_actual, cell_expected)
        assert all(atoms.pbc == 3 * [True])


@pytest.mark.parametrize("fileformat", (None, ".xyz"))
def test_write_predictions(fileformat, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    systems, predictions = systems_predictions()

    filename = "test_output.xyz"

    write_predictions(filename, predictions, systems, fileformat=fileformat)

    frames = ase.io.read(filename, index=":")
    assert len(frames) == len(systems)
    for i, frame in enumerate(frames):
        assert frame.info["energy"] == float(predictions["energy"].block().values[i, 0])


def test_write_predictions_unknown_fileformat():
    with pytest.raises(ValueError, match="fileformat '.bar' is not supported"):
        write_predictions("foo.bar", predictions=None, systems=None)
