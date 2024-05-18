import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.atomistic import System

from metatensor.models.utils.per_atom import (
    average_predictions_and_targets_by_num_atoms,
    divide_by_num_atoms,
)


def test_average_predictions_and_targets_by_num_atoms():
    """Tests the average_predictions_and_targets_by_num_atoms function."""

    systems = [
        System(
            positions=torch.tensor([[0.0, 0.0, 0.0]]),
            cell=torch.eye(3),
            types=torch.tensor([0]),
        ),
        System(
            positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            cell=torch.eye(3),
            types=torch.tensor([0, 0]),
        ),
        System(
            positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            cell=torch.eye(3),
            types=torch.tensor([0, 0, 0]),
        ),
    ]

    block = TensorBlock(
        values=torch.tensor([[1.0], [2.0], [3.0]]),
        samples=Labels.range("samples", 3),
        components=[],
        properties=Labels("energy", torch.tensor([[0]])),
    )

    tensor_map = TensorMap(keys=Labels.single(), blocks=[block])
    tensor_map_dict = {"energy": tensor_map}

    averaged_predictions, averaged_targets = (
        average_predictions_and_targets_by_num_atoms(
            tensor_map_dict, tensor_map_dict, systems, per_structure_targets=[]
        )
    )

    assert torch.allclose(
        averaged_predictions["energy"].block().values, torch.tensor([1.0, 1.0, 1.0])
    )
    assert torch.allclose(
        averaged_targets["energy"].block().values, torch.tensor([1.0, 1.0, 1.0])
    )


def test_divide_by_num_atoms():
    """Tests the divide_by_num_atoms function."""

    n_atoms = torch.tensor([1, 2, 3])

    block = TensorBlock(
        values=torch.tensor([[1.0], [2.0], [3.0]]),
        samples=Labels.range("samples", 3),
        components=[],
        properties=Labels("energy", torch.tensor([[0]])),
    )

    block.add_gradient(
        "position",
        TensorBlock(
            values=torch.tensor([[1.0], [2.0], [3.0]]),
            samples=Labels(
                names=["sample", "atom"],
                values=torch.stack(
                    [
                        torch.tensor([0, 1, 2]),
                        torch.tensor([0, 0, 0]),
                    ],
                    dim=1,
                ),
            ),
            components=[],
            properties=Labels("energy", torch.tensor([[0]])),
        ),
    )
    block.add_gradient(
        "strain",
        TensorBlock(
            values=torch.tensor([[1.0], [2.0], [3.0]]),
            samples=Labels.range("sample", 3),
            components=[],
            properties=Labels("energy", torch.tensor([[0]])),
        ),
    )

    tensor_map = TensorMap(keys=Labels.single(), blocks=[block])

    tensor_map = divide_by_num_atoms(tensor_map, n_atoms)

    # energies and virials should be divided by the number of atoms
    assert torch.allclose(tensor_map.block().values, torch.tensor([1.0, 1.0, 1.0]))
    assert torch.allclose(
        tensor_map.block().gradient("strain").values,
        torch.tensor([[1.0], [1.0], [1.0]]),
    )

    # forces should not be
    assert torch.allclose(
        tensor_map.block().gradient("position").values,
        torch.tensor([[1.0], [2.0], [3.0]]),
    )
