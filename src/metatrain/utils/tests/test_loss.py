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


def _vector_mask_tensormap(mask: torch.Tensor, systems: torch.Tensor) -> TensorMap:
    block = TensorBlock(
        values=mask.reshape(mask.shape[0], mask.shape[1], 1),
        samples=Labels(
            names=["system", "atom"],
            values=torch.stack(
                [systems, torch.arange(mask.shape[0], dtype=torch.int64)], dim=1
            ),
        ),
        components=[Labels.range("o3_mu", mask.shape[1])],
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


def test_invariant_losses_skip_entities_with_missing_target_components() -> None:
    prediction = _vector_tensormap(
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        torch.tensor([0, 0], dtype=torch.int64),
    )
    target = _vector_tensormap(
        torch.tensor([[0.0, 0.0, 0.0], [0.0, float("nan"), 0.0]]),
        torch.tensor([0, 0], dtype=torch.int64),
    )

    loss = create_loss(
        "invariant_mse",
        name="forces",
        gradient=None,
        weight=1.0,
        reduction="mean",
    )

    value = loss({"forces": prediction}, {"forces": target})

    assert torch.isclose(value, torch.tensor(14.0 / 3.0))


def test_invariant_losses_respect_masks_for_supported_components() -> None:
    systems = torch.tensor([0], dtype=torch.int64)
    prediction = _vector_tensormap(torch.tensor([[3.0, 4.0, 10.0]]), systems)
    target = _vector_tensormap(torch.tensor([[0.0, 0.0, float("nan")]]), systems)
    mask = _vector_mask_tensormap(torch.tensor([[True, True, False]]), systems)

    loss = create_loss(
        "invariant_mse",
        name="forces",
        gradient=None,
        weight=1.0,
        reduction="mean",
    )

    value = loss({"forces": prediction}, {"forces": target}, {"forces_mask": mask})

    assert torch.isclose(value, torch.tensor(12.5))


def test_invariant_losses_do_not_silently_drop_supported_nonfinite_predictions() -> None:
    prediction = _vector_tensormap(
        torch.tensor([[float("nan"), 2.0, 3.0]]),
        torch.tensor([0], dtype=torch.int64),
    )
    target = _vector_tensormap(
        torch.tensor([[0.0, 0.0, 0.0]]),
        torch.tensor([0], dtype=torch.int64),
    )

    loss = create_loss(
        "invariant_mse",
        name="forces",
        gradient=None,
        weight=1.0,
        reduction="mean",
    )

    value = loss({"forces": prediction}, {"forces": target})

    assert torch.isnan(value)


def test_invariant_huber_has_finite_zero_residual_gradients() -> None:
    parameter = torch.tensor(0.0, requires_grad=True)
    prediction = _energy_with_positions_gradient(
        energy_values=torch.tensor([0.0]),
        gradient_values=parameter.reshape(1, 1).expand(1, 3),
        gradient_samples=torch.tensor([[0, 0]], dtype=torch.int64),
    )
    target = _energy_with_positions_gradient(
        energy_values=torch.tensor([0.0]),
        gradient_values=torch.zeros((1, 3)),
        gradient_samples=torch.tensor([[0, 0]], dtype=torch.int64),
    )

    loss = create_loss(
        "invariant_huber",
        name="energy",
        gradient="positions",
        weight=1.0,
        reduction="mean",
        delta=0.04,
    )

    value = loss({"energy": prediction}, {"energy": target})
    value.backward()

    assert torch.isfinite(value)
    assert parameter.grad is not None
    assert torch.isfinite(parameter.grad)
