import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from metatrain.utils.loss import create_loss


def _vector_tensormap(values: torch.Tensor, systems: torch.Tensor) -> TensorMap:
    block = TensorBlock(
        values=values.reshape(values.shape[0], values.shape[1], 1),
        samples=Labels(
            names=["system", "atom"],
            values=torch.stack(
                [systems, torch.arange(values.shape[0], dtype=torch.int64)], dim=1
            ),
        ),
        components=[Labels.range("o3_mu", values.shape[1])],
        properties=Labels.range("property", 1),
    )
    return TensorMap(keys=Labels.single(), blocks=[block])


def _energy_with_positions_gradient(
    energy_values: torch.Tensor,
    gradient_values: torch.Tensor,
    gradient_samples: torch.Tensor,
) -> TensorMap:
    block = TensorBlock(
        values=energy_values.reshape(energy_values.shape[0], 1),
        samples=Labels.range("system", energy_values.shape[0]),
        components=[],
        properties=Labels.range("property", 1),
    )
    block.add_gradient(
        "positions",
        TensorBlock(
            values=gradient_values.reshape(
                gradient_values.shape[0], gradient_values.shape[1], 1
            ),
            samples=Labels(
                names=["sample", "atom"],
                values=gradient_samples,
            ),
            components=[Labels.range("xyz", gradient_values.shape[1])],
            properties=Labels.range("property", 1),
        ),
    )
    return TensorMap(keys=Labels.single(), blocks=[block])


def test_invariant_huber_is_rotation_invariant_for_vector_targets() -> None:
    prediction = _vector_tensormap(
        torch.tensor([[1.0, 2.0, 3.0], [0.5, -1.0, 1.5]]),
        torch.tensor([0, 0], dtype=torch.int64),
    )
    target = _vector_tensormap(
        torch.tensor([[0.0, 1.0, 1.0], [1.0, 0.0, 0.0]]),
        torch.tensor([0, 0], dtype=torch.int64),
    )

    loss = create_loss(
        "invariant_huber",
        name="forces",
        gradient=None,
        weight=1.0,
        reduction="mean",
        delta=0.5,
    )

    base_loss = loss({"forces": prediction}, {"forces": target})

    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=prediction.block().values.dtype,
    )
    rotated_prediction = _vector_tensormap(
        torch.einsum("ij,sj->si", rotation, prediction.block().values[..., 0]),
        torch.tensor([0, 0], dtype=torch.int64),
    )
    rotated_target = _vector_tensormap(
        torch.einsum("ij,sj->si", rotation, target.block().values[..., 0]),
        torch.tensor([0, 0], dtype=torch.int64),
    )

    rotated_loss = loss({"forces": rotated_prediction}, {"forces": rotated_target})

    assert torch.allclose(base_loss, rotated_loss)


def test_invariant_mse_uses_structure_first_averaging() -> None:
    prediction = _vector_tensormap(
        torch.tensor([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0], [5.0, 5.0, 5.0]]),
        torch.tensor([0, 1, 1], dtype=torch.int64),
    )
    target = _vector_tensormap(
        torch.zeros((3, 3)),
        torch.tensor([0, 1, 1], dtype=torch.int64),
    )

    loss = create_loss(
        "invariant_mse",
        name="forces",
        gradient=None,
        weight=1.0,
        reduction="mean",
    )

    value = loss({"forces": prediction}, {"forces": target})

    assert torch.isclose(value, torch.tensor(9.0))


def test_invariant_losses_support_gradient_rows() -> None:
    prediction = _energy_with_positions_gradient(
        energy_values=torch.tensor([0.0, 0.0]),
        gradient_values=torch.tensor(
            [
                [3.0, 4.0, 0.0],
                [0.0, 0.0, 0.0],
                [6.0, 8.0, 0.0],
            ]
        ),
        gradient_samples=torch.tensor([[0, 0], [1, 0], [1, 1]], dtype=torch.int64),
    )
    target = _energy_with_positions_gradient(
        energy_values=torch.tensor([0.0, 0.0]),
        gradient_values=torch.zeros((3, 3)),
        gradient_samples=torch.tensor([[0, 0], [1, 0], [1, 1]], dtype=torch.int64),
    )

    loss = create_loss(
        "invariant_mse",
        name="energy",
        gradient="positions",
        weight=1.0,
        reduction="mean",
    )

    value = loss({"energy": prediction}, {"energy": target})

    assert torch.isclose(value, torch.tensor(12.5))
