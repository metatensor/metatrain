import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from metatrain.utils.metrics import MAEAccumulator, RMSEAccumulator, get_selected_metric


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

    rmses = rmse_accumulator.finalize(not_per_atom=["gradient_gradients"])

    assert "energy RMSE (per atom)" in rmses
    assert "energy_gradient_gradients RMSE" in rmses


def test_mae_accumulator(tensor_map_with_grad_1, tensor_map_with_grad_2):
    """Tests the MAEAccumulator class."""

    mae_accumulator = MAEAccumulator()
    for _ in range(10):
        mae_accumulator.update(
            {"energy": tensor_map_with_grad_1}, {"energy": tensor_map_with_grad_2}
        )

    assert mae_accumulator.information["energy"][1] == 30
    assert mae_accumulator.information["energy_gradient_gradients"][1] == 30

    maes = mae_accumulator.finalize(not_per_atom=["gradient_gradients"])

    assert "energy MAE (per atom)" in maes
    assert "energy_gradient_gradients MAE" in maes


@pytest.mark.parametrize("accumulator_class", [RMSEAccumulator, MAEAccumulator])
def test_per_block(accumulator_class, tensor_map_with_grad_1, tensor_map_with_grad_2):
    """Tests that separate errors per block works."""

    tensor_1 = TensorMap(
        keys=Labels.range("label_name", 2),
        blocks=[tensor_map_with_grad_1.block(), tensor_map_with_grad_1.block()],
    )
    tensor_2 = TensorMap(
        keys=Labels.range("label_name", 2),
        blocks=[tensor_map_with_grad_2.block(), tensor_map_with_grad_2.block()],
    )

    accumulator = accumulator_class(separate_blocks=True)
    for _ in range(10):
        accumulator.update({"energy": tensor_1}, {"energy": tensor_2})

    assert accumulator.information["energy_label_name_0"][1] == 30
    assert accumulator.information["energy_label_name_1"][1] == 30
    assert accumulator.information["energy_label_name_0_gradient_gradients"][1] == 30
    assert accumulator.information["energy_label_name_1_gradient_gradients"][1] == 30

    metrics = accumulator.finalize(not_per_atom=["gradient_gradients"])

    rmse_or_mae = "RMSE" if accumulator_class == RMSEAccumulator else "MAE"
    assert f"energy(label_name=0) {rmse_or_mae} (per atom)" in metrics
    assert f"energy(label_name=0)_gradient_gradients {rmse_or_mae}" in metrics
    assert f"energy(label_name=1) {rmse_or_mae} (per atom)" in metrics
    assert f"energy(label_name=1)_gradient_gradients {rmse_or_mae}" in metrics


def test_get_selected_metric():
    """Tests the get_selected_metric function."""

    metrics = {
        "loss": 1,
        "energy RMSE": 2,
        "energy MAE": 3,
        "mtt::target RMSE": 4,
        "mtt::target MAE": 5,
    }

    selected_metric = "foo"
    with pytest.raises(ValueError, match="Please select from"):
        get_selected_metric(metrics, selected_metric)

    selected_metric = "rmse_prod"
    assert get_selected_metric(metrics, selected_metric) == 8

    selected_metric = "mae_prod"
    assert get_selected_metric(metrics, selected_metric) == 15

    selected_metric = "loss"
    assert get_selected_metric(metrics, selected_metric) == 1
