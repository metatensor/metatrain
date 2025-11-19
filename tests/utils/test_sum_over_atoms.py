import metatensor.torch as mts
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from metatrain.utils.sum_over_atoms import sum_over_atoms


def test_sum_over_atoms():
    """Test the sum_over_atoms function."""
    block1 = TensorBlock(
        values=torch.tensor([[[1.0]], [[2.0]], [[3.0]]]),
        samples=Labels(
            names=["system", "atom"],
            values=torch.tensor([[0, 0], [0, 1], [1, 0]]),
        ),
        components=[Labels.range("comp", 1)],
        properties=Labels.single(),
    )

    block2 = TensorBlock(
        values=torch.tensor([[[4.0], [5.0]], [[6.0], [7.0]], [[8.0], [9.0]]]),
        samples=Labels(
            names=["system", "atom"],
            values=torch.tensor([[0, 0], [0, 1], [1, 0]]),
        ),
        components=[Labels.range("comp", 2)],
        properties=Labels.single(),
    )

    tensor_map = TensorMap(
        keys=Labels.range("key", 2),
        blocks=[block1, block2],
    )

    # Call the sum_over_atoms function
    summed_tensor_map = sum_over_atoms(tensor_map)

    summed_tensor_map_ref = mts.sum_over_samples(
        tensor_map,
        sample_names=["atom"],
    )

    assert mts.allclose(summed_tensor_map, summed_tensor_map_ref)


def test_sum_over_atoms_empty():
    """Test sum_over_atoms with empty blocks or samples."""

    empty_tensor_map = TensorMap(keys=Labels.empty(["key"]), blocks=[])
    summed_empty = sum_over_atoms(empty_tensor_map)
    assert len(summed_empty.keys) == 0
    assert len(summed_empty.blocks()) == 0

    block_empty_samples = TensorBlock(
        values=torch.empty((0, 2, 1), dtype=torch.float64),
        samples=Labels.empty(["system", "atom"]),
        components=[Labels.range("comp", 2)],
        properties=Labels.single(),
    )
    tensor_map_empty_samples = TensorMap(
        keys=Labels.single(), blocks=[block_empty_samples]
    )
    summed_empty_samples = sum_over_atoms(tensor_map_empty_samples)

    assert len(summed_empty_samples.keys) == 1
    summed_block = summed_empty_samples.block()
    assert summed_block.values.shape == (0, 2, 1)
    assert summed_block.samples.names == ["system"]
    assert len(summed_block.samples) == 0
