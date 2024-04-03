import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from metatensor.models.utils.metrics import RMSEAccumulator


@pytest.fixture
def tensor_map_with_grad_1():
    block = TensorBlock(
        values=torch.tensor([[1.0], [2.0], [3.0]]),
        samples=Labels.range("samples", 3),
        components=[],
        properties=Labels("energy", torch.tensor([[0]])),
    )
    block.add_gradient(
        "gradient",
        TensorBlock(
            values=torch.tensor([[1.0], [2.0], [3.0]]),
            samples=Labels.range("sample", 3),
            components=[],
            properties=Labels("energy", torch.tensor([[0]])),
        ),
    )
    tensor_map = TensorMap(keys=Labels.single(), blocks=[block])
    return tensor_map


@pytest.fixture
def tensor_map_with_grad_2():
    block = TensorBlock(
        values=torch.tensor([[1.0], [1.0], [3.0]]),
        samples=Labels.range("samples", 3),
        components=[],
        properties=Labels("energy", torch.tensor([[0]])),
    )
    block.add_gradient(
        "gradient",
        TensorBlock(
            values=torch.tensor([[1.0], [0.0], [3.0]]),
            samples=Labels.range("sample", 3),
            components=[],
            properties=Labels("energy", torch.tensor([[0]])),
        ),
    )
    tensor_map = TensorMap(keys=Labels.single(), blocks=[block])
    return tensor_map


def test_rmse_accumulator(tensor_map_with_grad_1, tensor_map_with_grad_2):
    """Tests the RMSEAccumulator class."""

    rmse_accumulator = RMSEAccumulator()
    for _ in range(10):
        rmse_accumulator.update(
            {"energy": tensor_map_with_grad_1}, {"energy": tensor_map_with_grad_2}
        )

    assert rmse_accumulator.information["energy"][1] == 30
    assert rmse_accumulator.information["energy_gradient_gradients"][1] == 30

    rmses = rmse_accumulator.finalize()

    assert "energy RMSE" in rmses
    assert "energy_gradient_gradients RMSE" in rmses
