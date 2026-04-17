import numpy as np
import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from metatrain.utils.metrics import (
    MAEAccumulator,
    RMSEAccumulator,
    _get_global_keys,
    get_selected_metric,
)


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


def test_get_global_keys_raises():
    """
    Tests the _get_global_keys function raises an error if distributed is not available.
    """

    keys = ["a", "b", "c"]
    with pytest.raises(
        ValueError, match="Default process group has not been initialized"
    ):
        _get_global_keys(keys)


def test_get_global_keys_basic(monkeypatch):
    """
    Tests _get_global_keys unions, deduplicates and sorts keys across ranks.
    """

    def fake_get_world_size():
        return 2

    def fake_all_gather_object(output_list, local_obj):
        output_list[0] = local_obj
        output_list[1] = ["b", "c", "d"]

    monkeypatch.setattr(torch.distributed, "get_world_size", fake_get_world_size)
    monkeypatch.setattr(torch.distributed, "all_gather_object", fake_all_gather_object)

    result = _get_global_keys(["a", "b", "c"])
    assert result == ["a", "b", "c", "d"]


def test_rmse_finalize_distributed_missing_key_on_local_rank(monkeypatch):
    """
    Tests RMSEAccumulator.finalize with is_distributed=True when a key present on a
    remote rank is absent from the local rank. This is the core scenario fixed by the
    branch: the local rank must contribute zeros for that key so all_reduce produces
    the correct global RMSE.
    """

    # Rank 0 has only "energy"; rank 1 also has "forces"
    def fake_get_world_size():
        return 2

    def fake_all_gather_object(output_list, local_obj):
        output_list[0] = local_obj  # ["energy"]
        output_list[1] = ["energy", "forces"]

    # all_reduce is called once per (key, tensor) pair in sorted key order:
    # "energy" sse, "energy" n_elems, "forces" sse, "forces" n_elems
    remote_additions = [4.0, 2, 9.0, 3]
    call_count = {"i": 0}

    def fake_all_reduce(tensor):
        tensor.add_(remote_additions[call_count["i"]])
        call_count["i"] += 1

    monkeypatch.setattr(torch.distributed, "get_world_size", fake_get_world_size)
    monkeypatch.setattr(torch.distributed, "all_gather_object", fake_all_gather_object)
    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    acc = RMSEAccumulator()
    acc.information = {"energy": (4.0, 2)}  # sse=4.0, n_elems=2

    result = acc.finalize(
        not_per_atom=[], is_distributed=True, device=torch.device("cpu")
    )

    # "forces" must appear — regression guard for the bug
    assert "forces RMSE (per atom)" in result
    # energy: (4.0 + 4.0) / (2 + 2) -> sqrt(2.0)
    assert abs(result["energy RMSE (per atom)"] - 2.0**0.5) < 1e-6
    # forces: (0.0 + 9.0) / (0 + 3) -> sqrt(3.0)
    assert abs(result["forces RMSE (per atom)"] - 3.0**0.5) < 1e-6


@pytest.mark.parametrize("separate_blocks", [False, True])
def test_rmse_finalize_distributed_extra_key_only_on_local_rank(
    monkeypatch, separate_blocks
):
    """
    Tests RMSEAccumulator.finalize with is_distributed=True when the local rank has a
    key that the remote rank does not. The remote rank contributes zero via all_reduce.
    """

    def fake_get_world_size():
        return 2

    def fake_all_gather_object(output_list, local_obj):
        output_list[0] = local_obj  # ["energy", "forces"]
        output_list[1] = ["energy"]

    # Sorted global keys: ["energy", "forces"]
    # all_reduce call order: energy sse, energy n_elems, forces sse, forces n_elems
    # Remote rank contributes to "energy" but zero for "forces"
    remote_additions = [4.0, 2, 0.0, 0]
    call_count = {"i": 0}

    def fake_all_reduce(tensor):
        tensor.add_(remote_additions[call_count["i"]])
        call_count["i"] += 1

    monkeypatch.setattr(torch.distributed, "get_world_size", fake_get_world_size)
    monkeypatch.setattr(torch.distributed, "all_gather_object", fake_all_gather_object)
    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    acc = RMSEAccumulator(separate_blocks=separate_blocks)
    acc.information = {"energy": (4.0, 2), "forces": (9.0, 3)}

    result = acc.finalize(
        not_per_atom=[], is_distributed=True, device=torch.device("cpu")
    )

    # energy: (4.0 + 4.0) / (2 + 2) -> sqrt(2.0)
    assert abs(result["energy RMSE (per atom)"] - 2.0**0.5) < 1e-6
    # forces: (9.0 + 0.0) / (3 + 0) -> sqrt(3.0)
    assert abs(result["forces RMSE (per atom)"] - 3.0**0.5) < 1e-6


def test_mae_finalize_distributed_missing_key_on_local_rank(monkeypatch):
    """
    Tests MAEAccumulator.finalize with is_distributed=True when a key present on a
    remote rank is absent from the local rank.
    """

    def fake_get_world_size():
        return 2

    def fake_all_gather_object(output_list, local_obj):
        output_list[0] = local_obj  # ["energy"]
        output_list[1] = ["energy", "forces"]

    # Sorted global keys: ["energy", "forces"]
    # all_reduce call order: energy sae, energy n_elems, forces sae, forces n_elems
    remote_additions = [4.0, 4, 6.0, 3]
    call_count = {"i": 0}

    def fake_all_reduce(tensor):
        tensor.add_(remote_additions[call_count["i"]])
        call_count["i"] += 1

    monkeypatch.setattr(torch.distributed, "get_world_size", fake_get_world_size)
    monkeypatch.setattr(torch.distributed, "all_gather_object", fake_all_gather_object)
    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    acc = MAEAccumulator()
    acc.information = {"energy": (4.0, 4)}  # sae=4.0, n_elems=4

    result = acc.finalize(
        not_per_atom=[], is_distributed=True, device=torch.device("cpu")
    )

    # "forces" must appear — regression guard for the bug
    assert "forces MAE (per atom)" in result
    # energy: (4.0 + 4.0) / (4 + 4) = 1.0
    assert abs(result["energy MAE (per atom)"] - 1.0) < 1e-6
    # forces: (0.0 + 6.0) / (0 + 3) = 2.0
    assert abs(result["forces MAE (per atom)"] - 2.0) < 1e-6
