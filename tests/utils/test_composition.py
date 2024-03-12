from pathlib import Path

import torch
from metatensor.learn import Dataset
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.atomistic import System

from metatensor.models.utils.composition import calculate_composition_weights


RESOURCES_PATH = Path(__file__).parent.resolve() / ".." / "resources"


def test_calculate_composition_weights():
    """Test the calculation of composition weights."""

    # Here we use three synthetic structures:
    # - O atom, with an energy of 1.0
    # - H2O molecule, with an energy of 5.0
    # - H4O2 molecule, with an energy of 10.0
    # The expected composition weights are 2.0 for H and 1.0 for O.

    systems = [
        System(
            positions=torch.tensor([[0.0, 0.0, 0.0]]),
            types=torch.tensor([8]),
            cell=torch.eye(3),
        ),
        System(
            positions=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            types=torch.tensor([1, 1, 8]),
            cell=torch.eye(3),
        ),
        System(
            positions=torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [0.0, 1.0, 1.0],
                ]
            ),
            types=torch.tensor([1, 1, 8, 1, 1, 8]),
            cell=torch.eye(3),
        ),
    ]
    energies = [1.0, 5.0, 10.0]
    energies = [
        TensorMap(
            keys=Labels(names=["_"], values=torch.tensor([[0]])),
            blocks=[
                TensorBlock(
                    values=torch.tensor([[e]]),
                    samples=Labels(names=["system"], values=torch.tensor([[i]])),
                    components=[],
                    properties=Labels(names=["energy"], values=torch.tensor([[0]])),
                )
            ],
        )
        for i, e in enumerate(energies)
    ]
    dataset = Dataset(system=systems, energy=energies)

    weights, species = calculate_composition_weights(dataset, "energy")

    assert len(weights) == len(species)
    assert len(weights) == 2
    assert species == [1, 8]
    assert torch.allclose(weights, torch.tensor([2.0, 1.0]))
