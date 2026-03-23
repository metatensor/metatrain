import numpy as np
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
    block.add_gradient(
        "strain",
        TensorBlock(
            values=torch.tensor([[1.0], [np.nan], [3.0]]),
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
    block.add_gradient(
        "strain",
        TensorBlock(
            values=torch.tensor([[1.0], [np.nan], [2.0]]),
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
    assert rmse_accumulator.information["energy_strain_gradients"][1] == 20

    rmses = rmse_accumulator.finalize(not_per_atom=["gradient_gradients"])

    assert "energy RMSE (per atom)" in rmses
    assert "energy_gradient_gradients RMSE" in rmses
    assert "energy_strain_gradients RMSE (per atom)" in rmses


def test_mae_accumulator(tensor_map_with_grad_1, tensor_map_with_grad_2):
    """Tests the MAEAccumulator class."""

    mae_accumulator = MAEAccumulator()
    for _ in range(10):
        mae_accumulator.update(
            {"energy": tensor_map_with_grad_1}, {"energy": tensor_map_with_grad_2}
        )

    assert mae_accumulator.information["energy"][1] == 30
    assert mae_accumulator.information["energy_gradient_gradients"][1] == 30
    assert mae_accumulator.information["energy_strain_gradients"][1] == 20
    maes = mae_accumulator.finalize(not_per_atom=["gradient_gradients"])

    assert "energy MAE (per atom)" in maes
    assert "energy_gradient_gradients MAE" in maes
    assert "energy_strain_gradients MAE (per atom)" in maes


def test_rmse_accumulator_nan_stress(tensor_map_with_nan_1, tensor_map_with_nan_2):
    """Tests the RMSEAccumulator class with NaN values in non_conservative_stress."""

    rmse_accumulator = RMSEAccumulator()
    rmse_accumulator.update(
        {"non_conservative_stress": tensor_map_with_nan_1},
        {"non_conservative_stress": tensor_map_with_nan_2},
    )

    # Expected values for non_conservative_stress
    # predictions: [1.0, 2.0, NaN]
    # targets:     [1.0, 3.0, NaN]
    # Differences (non-NaN): [0.0, -1.0]
    # Squared differences: [0.0, 1.0]
    # SSE: 1.0
    # n_elems: 2
    # RMSE: sqrt(1.0 / 2) = sqrt(0.5) approx 0.70710678

    # Expected values for non_conservative_stress_strain_gradients
    # predictions: [1.0, NaN, 3.0]
    # targets:     [2.0, NaN, 4.0]
    # Differences (non-NaN): [-1.0, -1.0]
    # Squared differences: [1.0, 1.0]
    # SSE: 2.0
    # n_elems: 2
    # RMSE: sqrt(2.0 / 2) = sqrt(1.0) = 1.0

    rmses = rmse_accumulator.finalize(not_per_atom=["strain_gradients"])

    assert "non_conservative_stress RMSE (per atom)" in rmses
    assert "non_conservative_stress_strain_gradients RMSE" in rmses

    assert abs(rmses["non_conservative_stress RMSE (per atom)"] - (0.5**0.5)) < 1e-6
    assert abs(rmses["non_conservative_stress_strain_gradients RMSE"] - 1.0) < 1e-6


def test_mae_accumulator_nan_stress(tensor_map_with_nan_1, tensor_map_with_nan_2):
    """Tests the MAEAccumulator class with NaN values in non_conservative_stress."""

    mae_accumulator = MAEAccumulator()
    mae_accumulator.update(
        {"non_conservative_stress": tensor_map_with_nan_1},
        {"non_conservative_stress": tensor_map_with_nan_2},
    )

    # Expected values for non_conservative_stress
    # predictions: [1.0, 2.0, NaN]
    # targets:     [1.0, 3.0, NaN]
    # Differences (non-NaN): [0.0, -1.0]
    # Absolute differences: [0.0, 1.0]
    # SAE: 1.0
    # n_elems: 2
    # MAE: 1.0 / 2 = 0.5

    # Expected values for non_conservative_stress_strain_gradients
    # predictions: [1.0, NaN, 3.0]
    # targets:     [2.0, NaN, 4.0]
    # Differences (non-NaN): [-1.0, -1.0]
    # Absolute differences: [1.0, 1.0]
    # SAE: 2.0
    # n_elems: 2
    # MAE: 2.0 / 2 = 1.0

    maes = mae_accumulator.finalize(not_per_atom=["strain_gradients"])

    assert "non_conservative_stress MAE (per atom)" in maes
    assert "non_conservative_stress_strain_gradients MAE" in maes

    assert abs(maes["non_conservative_stress MAE (per atom)"] - 0.5) < 1e-6
    assert abs(maes["non_conservative_stress_strain_gradients MAE"] - 1.0) < 1e-6


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

    assert accumulator.information["energy (label_name=0)"][1] == 30
    assert accumulator.information["energy (label_name=1)"][1] == 30
    assert accumulator.information["energy (label_name=0)_gradient_gradients"][1] == 30
    assert accumulator.information["energy (label_name=1)_gradient_gradients"][1] == 30

    metrics = accumulator.finalize(not_per_atom=["gradient_gradients"])

    rmse_or_mae = "RMSE" if accumulator_class == RMSEAccumulator else "MAE"
    assert f"energy (label_name=0) {rmse_or_mae} (per atom)" in metrics
    assert f"energy (label_name=0)_gradient_gradients {rmse_or_mae}" in metrics
    assert f"energy (label_name=1) {rmse_or_mae} (per atom)" in metrics
    assert f"energy (label_name=1)_gradient_gradients {rmse_or_mae}" in metrics


@pytest.fixture
def tensor_map_with_nan_1():
    block = TensorBlock(
        values=torch.tensor([[1.0], [2.0], [np.nan]]),
        samples=Labels.range("samples", 3),
        components=[],
        properties=Labels("stress", torch.tensor([[0]])),
    )
    block.add_gradient(
        "strain",
        TensorBlock(
            values=torch.tensor([[1.0], [np.nan], [3.0]]),
            samples=Labels.range("sample", 3),
            components=[],
            properties=Labels("stress", torch.tensor([[0]])),
        ),
    )
    tensor_map = TensorMap(
        keys=Labels.single(),
        blocks=[block],
    )
    return tensor_map


@pytest.fixture
def tensor_map_with_nan_2():
    block = TensorBlock(
        values=torch.tensor([[1.0], [3.0], [np.nan]]),
        samples=Labels.range("samples", 3),
        components=[],
        properties=Labels("stress", torch.tensor([[0]])),
    )
    block.add_gradient(
        "strain",
        TensorBlock(
            values=torch.tensor([[2.0], [np.nan], [4.0]]),
            samples=Labels.range("sample", 3),
            components=[],
            properties=Labels("stress", torch.tensor([[0]])),
        ),
    )
    tensor_map = TensorMap(
        keys=Labels.single(),
        blocks=[block],
    )
    return tensor_map


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
