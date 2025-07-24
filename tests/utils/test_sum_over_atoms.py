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
