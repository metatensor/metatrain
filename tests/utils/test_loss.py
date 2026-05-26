# tests/test_losses.py

import math
from pathlib import Path

import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from metatrain.utils.data import TargetInfo
from metatrain.utils.loss import (
    GaussianCRPSLoss,
    LossAggregator,
    LossType,
    DensityMSELossViaW,
    DensityMSELossViaC,
    TensorMapGaussianCRPSLoss,
    TensorMapGaussianNLLLoss,
    TensorMapHuberLoss,
    TensorMapMAELoss,
    TensorMapMaskedHuberLoss,
    TensorMapMaskedMAELoss,
    TensorMapMaskedMSELoss,
    TensorMapMSELoss,
    _ri_coefficients_delta_pyscf_order,
    _ri_coefficients_pyscf_order,
    create_loss,
)
from metatrain.utils.pyscf_loss import (
    overlap_matrix_name,
    pack_two_center_matrices,
    ri_density_fit_constant_name,
    ri_projections_name,
)


RESOURCES_PATH = Path(__file__).parents[1] / "resources"


@pytest.fixture
def tensor_map_with_grad_1():
    block = TensorBlock(
        values=torch.tensor([[1.0], [2.0], [3.0]]),
        samples=Labels.range("sample", 3),
        components=[],
        properties=Labels("energy", torch.tensor([[0]])),
    )
    block.add_gradient(
        "positions",
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
        samples=Labels.range("sample", 3),
        components=[],
        properties=Labels("energy", torch.tensor([[0]])),
    )
    block.add_gradient(
        "positions",
        TensorBlock(
            values=torch.tensor([[1.0], [0.0], [3.0]]),
            samples=Labels.range("sample", 3),
            components=[],
            properties=Labels("energy", torch.tensor([[0]])),
        ),
    )
    tensor_map = TensorMap(keys=Labels.single(), blocks=[block])
    return tensor_map


@pytest.fixture
def tensor_map_with_grad_3():
    block = TensorBlock(
        values=torch.tensor([[0.0], [1.0], [3.0]]),
        samples=Labels.range("sample", 3),
        components=[],
        properties=Labels("energy", torch.tensor([[0]])),
    )
    block.add_gradient(
        "positions",
        TensorBlock(
            values=torch.tensor([[1.0], [0.0], [3.0]]),
            samples=Labels.range("sample", 3),
            components=[],
            properties=Labels("energy", torch.tensor([[0]])),
        ),
    )
    tensor_map = TensorMap(keys=Labels.single(), blocks=[block])
    return tensor_map


@pytest.fixture
def tensor_map_with_grad_4():
    block = TensorBlock(
        values=torch.tensor([[0.0], [1.0], [3.0]]),
        samples=Labels.range("sample", 3),
        components=[],
        properties=Labels("energy", torch.tensor([[0]])),
    )
    block.add_gradient(
        "positions",
        TensorBlock(
            values=torch.tensor([[1.0], [0.0], [2.0]]),
            samples=Labels.range("sample", 3),
            components=[],
            properties=Labels("energy", torch.tensor([[0]])),
        ),
    )
    tensor_map = TensorMap(keys=Labels.single(), blocks=[block])
    return tensor_map


@pytest.fixture
def tensor_map_with_grad_1_with_strain():
    block = TensorBlock(
        values=torch.tensor([[0.0], [1.0], [3.0]]),
        samples=Labels.range("sample", 3),
        components=[],
        properties=Labels("energy", torch.tensor([[0]])),
    )
    block.add_gradient(
        "positions",
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
            values=torch.tensor([[1.0], [0.0], [3.0]]),
            samples=Labels.range("sample", 3),
            components=[],
            properties=Labels("energy", torch.tensor([[0]])),
        ),
    )
    tensor_map = TensorMap(keys=Labels.single(), blocks=[block])
    return tensor_map


@pytest.fixture
def tensor_map_with_grad_3_with_strain():
    block = TensorBlock(
        values=torch.tensor([[0.0], [1.0], [3.0]]),
        samples=Labels.range("sample", 3),
        components=[],
        properties=Labels("energy", torch.tensor([[0]])),
    )
    block.add_gradient(
        "positions",
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
            values=torch.tensor([[1.0], [0.0], [3.0]]),
            samples=Labels.range("sample", 3),
            components=[],
            properties=Labels("energy", torch.tensor([[0]])),
        ),
    )
    tensor_map = TensorMap(keys=Labels.single(), blocks=[block])
    return tensor_map


# Pointwise losses must return zero when predictions == targets
@pytest.mark.parametrize(
    "LossCls",
    [
        TensorMapMSELoss,
        TensorMapMAELoss,
        TensorMapHuberLoss,
    ],
)
def test_pointwise_zero_loss(tensor_map_with_grad_1, LossCls):
    tm = tensor_map_with_grad_1
    key = tm.keys.names[0]
    if LossCls == TensorMapHuberLoss:
        loss = LossCls(name=key, gradient=None, weight=1.0, reduction="mean", delta=1.0)
    else:
        loss = LossCls(name=key, gradient=None, weight=1.0, reduction="mean")
    pred = {key: tm}
    targ = {key: tm}
    assert loss(pred, targ).item() == pytest.approx(0.0)


# Masked losses must error if no mask is supplied
@pytest.mark.parametrize(
    "MaskedCls",
    [
        TensorMapMaskedMSELoss,
        TensorMapMaskedMAELoss,
        TensorMapMaskedHuberLoss,
    ],
)
def test_masked_loss_error_on_missing_mask(tensor_map_with_grad_1, MaskedCls):
    tm = tensor_map_with_grad_1
    key = tm.keys.names[0]
    if MaskedCls == TensorMapMaskedHuberLoss:
        loss = MaskedCls(
            name=key, gradient=None, weight=1.0, reduction="mean", delta=1.0
        )
    else:
        loss = MaskedCls(name=key, gradient=None, weight=1.0, reduction="mean")
    with pytest.raises(ValueError):
        loss({key: tm}, {key: tm})


# Functional test for masked MSE: only unmasked element contributes
def test_masked_mse_behavior(tensor_map_with_grad_1, tensor_map_with_grad_2):
    tm1 = tensor_map_with_grad_1
    tm2 = tensor_map_with_grad_2
    key = tm1.keys.names[0]

    # Construct a mask TensorMap: only index 1 is True
    mask_vals = torch.tensor([[False], [True], [False]], dtype=torch.bool)
    mask_block = TensorBlock(
        values=mask_vals,
        samples=tm1.block(0).samples,
        components=tm1.block(0).components,
        properties=tm1.block(0).properties,
    )
    mask_map = TensorMap(keys=tm1.keys, blocks=[mask_block])
    extra_data = {f"{key}_mask": mask_map}

    loss = TensorMapMaskedMSELoss(name=key, gradient=None, weight=1.0, reduction="mean")
    # Only element 1 contributes: (1-2)^2 = 1
    result = loss({key: tm2}, {key: tm1}, extra_data)
    assert result.item() == pytest.approx(1.0)


# Factory and enum resolution
def test_loss_type_and_factory():
    mapping = {
        "mse": TensorMapMSELoss,
        "mae": TensorMapMAELoss,
        "huber": TensorMapHuberLoss,
        "masked_mse": TensorMapMaskedMSELoss,
        "masked_mae": TensorMapMaskedMAELoss,
        "masked_huber": TensorMapMaskedHuberLoss,
        "density_mse_via_c": DensityMSELossViaC,
        "density_mse_via_w": DensityMSELossViaW,
    }
    for key, cls in mapping.items():
        # LossType.from_key should return enum with .key
        lt = LossType.from_key(key)
        assert lt.key == key
        # Factory should produce correct class
        extra_kwargs = {}
        if key == "huber" or key == "masked_huber":
            extra_kwargs = {"delta": 1.0}
        loss = create_loss(
            key,
            name="dummy",
            gradient=None,
            weight=1.0,
            reduction="mean",
            **extra_kwargs,
        )
        assert isinstance(loss, cls)

    # Invalid keys raise ValueError
    with pytest.raises(ValueError):
        LossType.from_key("invalid_key")
    with pytest.raises(ValueError):
        create_loss(
            "invalid_key",
            name="dummy",
            gradient=None,
            weight=1.0,
            reduction="mean",
        )


# Point-wise gradient-only
@pytest.mark.parametrize(
    "LossCls, expected",
    [
        (TensorMapMSELoss, 1 / 3),  # MSEGradient: one error squared -> 1/3
        (TensorMapMAELoss, 1 / 3),  # MAEGradient: one abs error -> 1/3
        (TensorMapHuberLoss, 1 / 6),  # HuberGradient: 0.5*1^2 /3 = 1/6
    ],
)
def test_pointwise_gradient_loss(
    tensor_map_with_grad_3, tensor_map_with_grad_4, LossCls, expected
):
    tm3 = tensor_map_with_grad_3
    tm4 = tensor_map_with_grad_4
    key = tm3.keys.names[0]
    # instantiate with gradient extraction
    if LossCls == TensorMapHuberLoss:
        loss = LossCls(
            name=key, gradient="positions", weight=1.0, reduction="mean", delta=1.0
        )
    else:
        loss = LossCls(name=key, gradient="positions", weight=1.0, reduction="mean")
    val = loss({key: tm3}, {key: tm4}).item()
    assert val == pytest.approx(expected)


def test_create_loss_invalid_kwargs():
    # Passing `foo` into an MSELoss constructor will cause
    # a TypeError inside create_loss, which should be caught
    # and re-raised with our custom message.
    with pytest.raises(TypeError) as exc:
        create_loss(
            "mse", name="dummy", gradient=None, weight=1.0, reduction="mean", foo=123
        )
    msg = str(exc.value)
    assert "Error constructing loss 'mse'" in msg
    assert (
        "foo" in msg
    )  # original constructor error should mention the unexpected 'foo'


def _make_ri_tensor_map_from_blocks(samples, values_l0, values_l1) -> TensorMap:
    values_l0 = (
        values_l0
        if isinstance(values_l0, torch.Tensor)
        else torch.tensor(values_l0, dtype=torch.float64)
    )
    values_l1 = (
        values_l1
        if isinstance(values_l1, torch.Tensor)
        else torch.tensor(values_l1, dtype=torch.float64)
    )

    return TensorMap(
        keys=Labels(
            names=["o3_lambda", "o3_sigma"],
            values=torch.tensor([[0, 1], [1, 1]], dtype=torch.int32),
        ),
        blocks=[
            TensorBlock(
                values=values_l0,
                samples=Labels(
                    names=["system", "atom"],
                    values=torch.tensor(samples, dtype=torch.int32),
                ),
                components=[
                    Labels(
                        names=["o3_mu"], values=torch.tensor([[0]], dtype=torch.int32)
                    )
                ],
                properties=Labels(
                    names=["n"],
                    values=torch.arange(values_l0.shape[-1], dtype=torch.int32).reshape(
                        -1, 1
                    ),
                ),
            ),
            TensorBlock(
                values=values_l1,
                samples=Labels(
                    names=["system", "atom"],
                    values=torch.tensor(samples, dtype=torch.int32),
                ),
                components=[
                    Labels(
                        names=["o3_mu"],
                        values=torch.tensor([[-1], [0], [1]], dtype=torch.int32),
                    )
                ],
                properties=Labels(
                    names=["n"],
                    values=torch.arange(values_l1.shape[-1], dtype=torch.int32).reshape(
                        -1, 1
                    ),
                ),
            ),
        ],
    )


def _make_ri_tensor_map(values_l0, values_l1) -> TensorMap:
    return _make_ri_tensor_map_from_blocks(
        samples=[[0, 0]],
        values_l0=torch.tensor(values_l0, dtype=torch.float64).reshape(1, 1, 1),
        values_l1=torch.tensor(values_l1, dtype=torch.float64).reshape(1, 3, 1),
    )


def _make_ri_tensor_map_from_pyscf_vector(
    values: list[float] | torch.Tensor,
) -> TensorMap:
    vector = (
        values
        if isinstance(values, torch.Tensor)
        else torch.tensor(values, dtype=torch.float64)
    )
    return _make_ri_tensor_map(
        [vector[0].item()],
        [vector[2].item(), vector[3].item(), vector[1].item()],
    )


def _make_multisystem_ri_tensor_map(
    values_l0: torch.Tensor, values_l1: torch.Tensor
) -> TensorMap:
    return _make_ri_tensor_map_from_blocks(
        samples=[[0, 0], [0, 1], [1, 0]],
        values_l0=values_l0,
        values_l1=values_l1,
    )


def _make_ri_delta_map(pred: TensorMap, targ: TensorMap) -> TensorMap:
    blocks = []
    for key in pred.keys:
        block_pred = pred.block(key)
        block_targ = targ.block(key)
        blocks.append(
            TensorBlock(
                values=block_pred.values - block_targ.values,
                samples=block_pred.samples,
                components=block_pred.components,
                properties=block_pred.properties,
            )
        )

    return TensorMap(pred.keys, blocks)


def _make_multisystem_ri_prediction_and_target() -> tuple[TensorMap, TensorMap]:
    pred_l0 = torch.tensor(
        [
            [[1.0, 10.0]],
            [[2.0, float("nan")]],
            [[3.0, 30.0]],
        ],
        dtype=torch.float64,
    )
    targ_l0 = torch.tensor(
        [
            [[0.5, 8.0]],
            [[1.5, float("nan")]],
            [[1.0, 29.0]],
        ],
        dtype=torch.float64,
    )

    pred_l1 = torch.tensor(
        [
            [[1.0, 11.0], [2.0, 12.0], [3.0, 13.0]],
            [[4.0, float("nan")], [5.0, float("nan")], [6.0, float("nan")]],
            [[7.0, 17.0], [8.0, 18.0], [9.0, 19.0]],
        ],
        dtype=torch.float64,
    )
    targ_l1 = torch.tensor(
        [
            [[0.5, 10.0], [1.0, 11.0], [1.5, 12.0]],
            [[2.0, float("nan")], [2.5, float("nan")], [3.0, float("nan")]],
            [[3.5, 16.0], [4.0, 17.0], [4.5, 18.0]],
        ],
        dtype=torch.float64,
    )

    pred = _make_multisystem_ri_tensor_map(pred_l0, pred_l1)
    targ = _make_multisystem_ri_tensor_map(targ_l0, targ_l1)
    return pred, targ


def _manual_unpadded_ri_deltas(pred: TensorMap, targ: TensorMap) -> list[torch.Tensor]:
    ordered_keys = sorted(pred.keys, key=lambda key: int(key[0]))
    if len(ordered_keys) == 0:
        return []

    first_block = pred.block(ordered_keys[0])
    if len(first_block.samples) == 0:
        return []

    system_ids = first_block.samples.values[:, 0].tolist()
    unique_system_ids = list(dict.fromkeys(system_ids))
    system_deltas = []

    for system_id in unique_system_ids:
        system_parts = []
        sample_mask = first_block.samples.values[:, 0] == system_id
        system_sample_indices = torch.nonzero(sample_mask, as_tuple=True)[0].tolist()

        for sample_index in system_sample_indices:
            for key in ordered_keys:
                block_pred = pred.block(key)
                block_targ = targ.block(key)
                delta = (
                    block_pred.values[sample_index] - block_targ.values[sample_index]
                )
                if int(key[0]) == 1:
                    delta = delta[[2, 0, 1], :]

                for radial_index in range(delta.shape[-1]):
                    for magnetic_index in range(delta.shape[0]):
                        value = delta[magnetic_index, radial_index]
                        if not torch.isnan(value):
                            system_parts.append(value.reshape(1))

        if len(system_parts) == 0:
            system_deltas.append(
                torch.empty(
                    0,
                    dtype=first_block.values.dtype,
                    device=first_block.values.device,
                )
            )
        else:
            system_deltas.append(torch.cat(system_parts))

    return system_deltas


def _manual_unpadded_ri_losses(
    pred: TensorMap, targ: TensorMap, matrices: list[torch.Tensor]
) -> torch.Tensor:
    deltas = _manual_unpadded_ri_deltas(pred, targ)
    return torch.stack(
        [
            torch.einsum("i,ij,j->", delta, matrix, delta)
            for delta, matrix in zip(deltas, matrices, strict=True)
        ]
    )


def _manual_unpadded_ri_values(tensor: TensorMap) -> list[torch.Tensor]:
    ordered_keys = sorted(tensor.keys, key=lambda key: int(key[0]))
    if len(ordered_keys) == 0:
        return []

    first_block = tensor.block(ordered_keys[0])
    if len(first_block.samples) == 0:
        return []

    system_ids = first_block.samples.values[:, 0].tolist()
    unique_system_ids = list(dict.fromkeys(system_ids))
    system_values = []

    for system_id in unique_system_ids:
        system_parts = []
        sample_mask = first_block.samples.values[:, 0] == system_id
        system_sample_indices = torch.nonzero(sample_mask, as_tuple=True)[0].tolist()

        for sample_index in system_sample_indices:
            for key in ordered_keys:
                block = tensor.block(key)
                values = block.values[sample_index]
                if int(key[0]) == 1:
                    values = values[[2, 0, 1], :]

                for radial_index in range(values.shape[-1]):
                    for magnetic_index in range(values.shape[0]):
                        value = values[magnetic_index, radial_index]
                        if not torch.isnan(value):
                            system_parts.append(value.reshape(1))

        if len(system_parts) == 0:
            system_values.append(
                torch.empty(
                    0,
                    dtype=first_block.values.dtype,
                    device=first_block.values.device,
                )
            )
        else:
            system_values.append(torch.cat(system_parts))

    return system_values


def test_ri_delta_helper_matches_reference_flattening():
    pred, targ = _make_multisystem_ri_prediction_and_target()

    reference = _manual_unpadded_ri_deltas(pred, targ)
    optimized = _ri_coefficients_delta_pyscf_order(pred, targ)

    assert len(optimized) == len(reference)
    for optimized_system, reference_system in zip(optimized, reference, strict=True):
        torch.testing.assert_close(optimized_system, reference_system)


def test_ri_coefficients_helper_matches_reference_flattening():
    tensor = _make_multisystem_ri_tensor_map(
        torch.tensor(
            [
                [[1.0, 10.0]],
                [[2.0, float("nan")]],
                [[3.0, 30.0]],
            ],
            dtype=torch.float64,
        ),
        torch.tensor(
            [
                [[1.0, 11.0], [2.0, 12.0], [3.0, 13.0]],
                [[4.0, float("nan")], [5.0, float("nan")], [6.0, float("nan")]],
                [[7.0, 17.0], [8.0, 18.0], [9.0, 19.0]],
            ],
            dtype=torch.float64,
        ),
    )

    reference = _manual_unpadded_ri_values(tensor)
    optimized = _ri_coefficients_pyscf_order(tensor)

    assert len(optimized) == len(reference)
    for optimized_system, reference_system in zip(optimized, reference, strict=True):
        torch.testing.assert_close(optimized_system, reference_system)



def test_density_overlap_loss_zero():
    target_name = "mtt::ri"
    tensor_map = _make_ri_tensor_map([1.0], [2.0, 3.0, 4.0])
    extra_data = {
        overlap_matrix_name(target_name): pack_two_center_matrices(
            [torch.eye(4, dtype=torch.float64)]
        )
    }

    loss = DensityMSELossViaC(
        name=target_name, gradient=None, weight=1.0, reduction="mean"
    )

    result = loss({target_name: tensor_map}, {target_name: tensor_map}, extra_data)
    assert result.item() == pytest.approx(0.0)


def test_density_overlap_loss_reorders_p_orbitals_for_pyscf():
    target_name = "mtt::ri"
    pred = _make_ri_tensor_map([0.0], [10.0, 20.0, 30.0])
    targ = _make_ri_tensor_map([0.0], [1.0, 2.0, 3.0])

    delta_pyscf = torch.tensor([0.0, 27.0, 9.0, 18.0], dtype=torch.float64)
    overlap = torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64))
    expected = torch.einsum("i,ij,j->", delta_pyscf, overlap, delta_pyscf)

    extra_data = {overlap_matrix_name(target_name): pack_two_center_matrices([overlap])}
    loss = DensityMSELossViaC(
        name=target_name, gradient=None, weight=1.0, reduction="mean"
    )

    result = loss({target_name: pred}, {target_name: targ}, extra_data)
    torch.testing.assert_close(result, expected)


def test_density_overlap_loss_matches_reference_multisystem():
    target_name = "mtt::ri"
    pred, targ = _make_multisystem_ri_prediction_and_target()
    manual_delta = _manual_unpadded_ri_deltas(pred, targ)

    matrices = []
    for coefficients in manual_delta:
        diag = torch.arange(1, len(coefficients) + 1, dtype=torch.float64)
        matrices.append(torch.diag(diag))

    expected = _manual_unpadded_ri_losses(pred, targ, matrices).mean()

    extra_data = {overlap_matrix_name(target_name): pack_two_center_matrices(matrices)}
    loss = DensityMSELossViaC(
        name=target_name, gradient=None, weight=1.0, reduction="mean"
    )

    result = loss({target_name: pred}, {target_name: targ}, extra_data)
    torch.testing.assert_close(result, expected)


def test_density_overlap_loss_reduction_none_matches_reference_multisystem():
    target_name = "mtt::ri"
    pred, targ = _make_multisystem_ri_prediction_and_target()
    manual_delta = _manual_unpadded_ri_deltas(pred, targ)

    matrices = []
    for coefficients in manual_delta:
        diag = torch.arange(1, len(coefficients) + 1, dtype=torch.float64)
        matrices.append(torch.diag(diag))

    expected = _manual_unpadded_ri_losses(pred, targ, matrices)

    extra_data = {overlap_matrix_name(target_name): pack_two_center_matrices(matrices)}
    loss = DensityMSELossViaC(
        name=target_name, gradient=None, weight=1.0, reduction="none"
    )

    result = loss({target_name: pred}, {target_name: targ}, extra_data)
    torch.testing.assert_close(result, expected)


def test_density_overlap_loss_gradient_matches_reference():
    target_name = "mtt::ri"

    pred_l0_new = torch.tensor(
        [
            [[1.0, 10.0]],
            [[2.0, float("nan")]],
            [[3.0, 30.0]],
        ],
        dtype=torch.float64,
    ).requires_grad_()
    pred_l1_new = torch.tensor(
        [
            [[1.0, 11.0], [2.0, 12.0], [3.0, 13.0]],
            [[4.0, float("nan")], [5.0, float("nan")], [6.0, float("nan")]],
            [[7.0, 17.0], [8.0, 18.0], [9.0, 19.0]],
        ],
        dtype=torch.float64,
    ).requires_grad_()
    pred_new = _make_multisystem_ri_tensor_map(pred_l0_new, pred_l1_new)

    pred_l0_ref = pred_l0_new.detach().clone().requires_grad_()
    pred_l1_ref = pred_l1_new.detach().clone().requires_grad_()
    pred_ref = _make_multisystem_ri_tensor_map(pred_l0_ref, pred_l1_ref)

    _, targ = _make_multisystem_ri_prediction_and_target()

    reference_delta = _manual_unpadded_ri_deltas(pred_ref, targ)
    matrices = []
    for coefficients in reference_delta:
        diag = torch.arange(1, len(coefficients) + 1, dtype=torch.float64)
        matrices.append(torch.diag(diag))

    extra_data = {overlap_matrix_name(target_name): pack_two_center_matrices(matrices)}
    loss = DensityMSELossViaC(
        name=target_name, gradient=None, weight=1.0, reduction="mean"
    )

    result = loss({target_name: pred_new}, {target_name: targ}, extra_data)
    result.backward()

    reference = _manual_unpadded_ri_losses(pred_ref, targ, matrices).mean()
    reference.backward()

    torch.testing.assert_close(pred_l0_new.grad, pred_l0_ref.grad, equal_nan=True)
    torch.testing.assert_close(pred_l1_new.grad, pred_l1_ref.grad, equal_nan=True)


def test_density_overlap_loss_reduction_none():
    target_name = "mtt::ri"
    pred = _make_ri_tensor_map([0.0], [1.0, 2.0, 3.0])
    targ = _make_ri_tensor_map([0.0], [0.0, 0.0, 0.0])
    overlap = torch.eye(4, dtype=torch.float64)
    extra_data = {overlap_matrix_name(target_name): pack_two_center_matrices([overlap])}

    loss = DensityMSELossViaC(
        name=target_name, gradient=None, weight=1.0, reduction="none"
    )

    result = loss({target_name: pred}, {target_name: targ}, extra_data)
    assert result.shape == (1,)


def test_density_overlap_loss_rejects_basis_mismatch():
    target_name = "mtt::ri"
    pred = _make_ri_tensor_map([0.0], [1.0, 2.0, 3.0])
    targ = _make_ri_tensor_map([0.0], [0.0, 0.0, 0.0])
    extra_data = {
        overlap_matrix_name(target_name): pack_two_center_matrices(
            [torch.eye(5, dtype=torch.float64)]
        )
    }

    loss = DensityMSELossViaC(
        name=target_name, gradient=None, weight=1.0, reduction="mean"
    )

    with pytest.raises(ValueError, match="RI target size does not match"):
        loss({target_name: pred}, {target_name: targ}, extra_data)


def test_density_fit_loss_matches_closed_form():
    target_name = "mtt::ri"
    pred = _make_ri_tensor_map([1.0], [2.0, 3.0, 4.0])
    targ = _make_ri_tensor_map([5.0], [6.0, 7.0, 8.0])

    c_pyscf = torch.tensor([1.0, 4.0, 2.0, 3.0], dtype=torch.float64)
    p_pyscf = torch.tensor([7.0, 17.0, 11.0, 13.0], dtype=torch.float64)
    overlap = torch.tensor(
        [
            [2.0, 0.1, 0.0, 0.0],
            [0.1, 3.0, 0.2, 0.0],
            [0.0, 0.2, 4.0, 0.3],
            [0.0, 0.0, 0.3, 5.0],
        ],
        dtype=torch.float64,
    )
    expected = c_pyscf @ overlap @ c_pyscf - 2.0 * c_pyscf @ p_pyscf

    extra_data = {
        overlap_matrix_name(target_name): pack_two_center_matrices([overlap]),
        ri_projections_name(target_name): _make_ri_tensor_map_from_pyscf_vector(p_pyscf),
    }
    loss = DensityMSELossViaW(
        name=target_name,
        gradient=None,
        weight=1.0,
        reduction="mean",
    )
    result = loss({target_name: pred}, {target_name: targ}, extra_data)
    torch.testing.assert_close(result, expected)


def test_density_fit_loss_minimum_at_c_true():
    """The density-fit loss is minimised at the true coefficients when the
    pre-computed constant c_RI^T w_RI is included in extra_data."""

    target_name = "mtt::ri"
    c_true = _make_ri_tensor_map([0.7], [1.5, -0.3, 2.1])

    c_pyscf = torch.tensor([0.7, 2.1, 1.5, -0.3], dtype=torch.float64)
    matrix = torch.tensor(
        [
            [2.0, 0.1, 0.05, 0.0],
            [0.1, 3.0, 0.2, 0.04],
            [0.05, 0.2, 1.5, 0.1],
            [0.0, 0.04, 0.1, 2.5],
        ],
        dtype=torch.float64,
    )
    p_pyscf = matrix @ c_pyscf  # w_RI = S c_RI

    from metatensor.torch import Labels, TensorBlock, TensorMap

    const_val = float(c_pyscf @ p_pyscf)
    const_block = TensorBlock(
        values=torch.tensor([[const_val]], dtype=torch.float64),
        samples=Labels(
            names=["system"],
            values=torch.tensor([[0]], dtype=torch.int32),
        ),
        components=[],
        properties=Labels(
            names=["_"],
            values=torch.zeros((1, 1), dtype=torch.int32),
        ),
    )
    const_map = TensorMap(Labels.single(), [const_block])

    extra_data = {
        overlap_matrix_name(target_name): pack_two_center_matrices([matrix]),
        ri_projections_name(target_name): _make_ri_tensor_map_from_pyscf_vector(p_pyscf),
        ri_density_fit_constant_name(target_name): const_map,
    }
    loss = DensityMSELossViaW(
        name=target_name,
        gradient=None,
        weight=1.0,
        reduction="mean",
    )

    result = loss({target_name: c_true}, {target_name: c_true}, extra_data)
    torch.testing.assert_close(result, torch.tensor(0.0, dtype=torch.float64), atol=1e-12, rtol=0)


def test_density_fit_loss_adds_constant_when_present():
    """Providing ri_density_fit_constant in extra_data shifts the loss by that value."""
    target_name = "mtt::ri"
    pred = _make_ri_tensor_map([0.5], [1.0, 2.0, 3.0])
    targ = _make_ri_tensor_map([0.0], [0.0, 0.0, 0.0])
    proj_pyscf = torch.tensor([0.4, 30.0, 10.0, 20.0], dtype=torch.float64)
    overlap = torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64))

    from metatensor.torch import Labels, TensorBlock, TensorMap

    constant_value = 42.0
    const_block = TensorBlock(
        values=torch.tensor([[constant_value]], dtype=torch.float64),
        samples=Labels(
            names=["system"],
            values=torch.tensor([[0]], dtype=torch.int32),
        ),
        components=[],
        properties=Labels(
            names=["_"],
            values=torch.zeros((1, 1), dtype=torch.int32),
        ),
    )
    const_map = TensorMap(Labels.single(), [const_block])

    extra_without = {
        overlap_matrix_name(target_name): pack_two_center_matrices([overlap]),
        ri_projections_name(target_name): _make_ri_tensor_map_from_pyscf_vector(proj_pyscf),
    }
    extra_with = {**extra_without, ri_density_fit_constant_name(target_name): const_map}

    loss = DensityMSELossViaW(
        name=target_name, gradient=None, weight=1.0, reduction="mean"
    )

    result_without = loss({target_name: pred}, {target_name: targ}, extra_without)
    result_with    = loss({target_name: pred}, {target_name: targ}, extra_with)
    torch.testing.assert_close(result_with - result_without,
                                torch.tensor(constant_value, dtype=torch.float64))


def test_density_fit_loss_requires_projection_extra_data():
    target_name = "mtt::ri"
    pred = _make_ri_tensor_map([0.0], [1.0, 2.0, 3.0])
    targ = _make_ri_tensor_map([0.0], [0.0, 0.0, 0.0])
    matrix = torch.eye(4, dtype=torch.float64)

    extra_data_without_projection = {
        overlap_matrix_name(target_name): pack_two_center_matrices([matrix])
    }
    loss = DensityMSELossViaW(
        name=target_name,
        gradient=None,
        weight=1.0,
        reduction="mean",
    )
    with pytest.raises(AssertionError, match=ri_projections_name(target_name)):
        loss({target_name: pred}, {target_name: targ}, extra_data_without_projection)


def test_density_fit_loss_requires_overlap_extra_data():
    """Density-fit loss must look up the overlap key."""
    target_name = "mtt::ri"
    pred = _make_ri_tensor_map([0.0], [1.0, 2.0, 3.0])
    targ = _make_ri_tensor_map([0.0], [0.0, 0.0, 0.0])
    matrix = torch.eye(4, dtype=torch.float64)

    only_projection = {
        ri_projections_name(target_name): _make_ri_tensor_map([0.0], [0.0, 0.0, 0.0]),
    }
    loss = DensityMSELossViaW(
        name=target_name,
        gradient=None,
        weight=1.0,
        reduction="mean",
    )
    with pytest.raises(AssertionError, match=overlap_matrix_name(target_name)):
        loss({target_name: pred}, {target_name: targ}, only_projection)


def test_density_fit_loss_rejects_basis_mismatch():
    target_name = "mtt::ri"
    pred = _make_ri_tensor_map([0.0], [1.0, 2.0, 3.0])
    targ = _make_ri_tensor_map([0.0], [0.0, 0.0, 0.0])
    extra_data = {
        overlap_matrix_name(target_name): pack_two_center_matrices(
            [torch.eye(5, dtype=torch.float64)]
        ),
        ri_projections_name(target_name): _make_ri_tensor_map([0.0], [0.0, 0.0, 0.0]),
    }

    loss = DensityMSELossViaW(
        name=target_name,
        gradient=None,
        weight=1.0,
        reduction="mean",
    )

    with pytest.raises(ValueError, match="RI target size does not match"):
        loss({target_name: pred}, {target_name: targ}, extra_data)


def test_masked_pointwise_gradient_branch(
    tensor_map_with_grad_3, tensor_map_with_grad_4
):
    tm3 = tensor_map_with_grad_3
    tm4 = tensor_map_with_grad_4
    key = tm3.keys.names[0]

    # Build a mask that selects all entries
    mask_vals = torch.tensor([[True], [True], [True]], dtype=torch.bool)
    mask_block = TensorBlock(
        values=mask_vals,
        samples=tm3.block(0).samples,
        components=tm3.block(0).components,
        properties=tm3.block(0).properties,
    )

    # Add a gradient-block to the mask, so grab(mask_block, "gradient") works
    grad_block_for_mask = TensorBlock(
        values=mask_vals,
        samples=tm3.block(0).samples,
        components=tm3.block(0).components,
        properties=tm3.block(0).properties,
    )
    mask_block.add_gradient("positions", grad_block_for_mask)

    mask_map = TensorMap(keys=tm3.keys, blocks=[mask_block])
    extra = {f"{key}_mask": mask_map}

    # Create the masked-pointwise loss on the 'positions' channel
    loss = TensorMapMaskedMSELoss(
        name=key, gradient="positions", weight=1.0, reduction="mean"
    )

    # The gradient values in tm3: [1, 0, 3]; in tm4: [1, 0, 2]
    # Only one difference of 1 -> MSE mean = 1/3
    result = loss({key: tm3}, {key: tm4}, extra).item()
    assert result == pytest.approx(1 / 3)


def test_tmap_loss_subset(tensor_map_with_grad_1, tensor_map_with_grad_3):
    """Test that the loss is computed correctly when only a subset
    of the possible targets is present both in outputs and targets."""

    block = TensorBlock(
        values=torch.empty(0, 1),
        samples=Labels(
            names=["system"],
            values=torch.empty((0, 1), dtype=torch.int32),
        ),
        components=[],
        properties=Labels.range("property", 1),
    )
    block.add_gradient(
        "positions",
        TensorBlock(
            values=torch.empty(0, 1),
            samples=Labels(
                names=["sample"],
                values=torch.empty((0, 1), dtype=torch.int32),
            ),
            components=[],
            properties=Labels.range("property", 1),
        ),
    )
    layout = TensorMap(keys=Labels.single(), blocks=[block])

    target_info = TargetInfo(layout=layout, quantity="energy", unit="eV")
    loss_hypers = {
        "output_1": {
            "type": "mse",
            "weight": 1.0,
            "reduction": "sum",
            "gradients": {
                "positions": {
                    "type": "mse",
                    "weight": 0.5,
                    "reduction": "sum",
                },
            },
        },
        "output_2": {
            "type": "mse",
            "weight": 1.0,
            "reduction": "sum",
            "gradients": {
                "positions": {
                    "type": "mse",
                    "weight": 0.5,
                    "reduction": "sum",
                },
            },
        },
    }

    loss = LossAggregator(
        targets={"output_1": target_info, "output_2": target_info},
        config=loss_hypers,
    )

    output_dict = {
        "output_1": tensor_map_with_grad_1,
    }

    target_dict = {
        "output_1": tensor_map_with_grad_3,
    }

    expected_result = (
        1.0
        * (
            tensor_map_with_grad_1.block().values
            - tensor_map_with_grad_3.block().values
        )
        .pow(2)
        .sum()
        + 0.5
        * (
            tensor_map_with_grad_1.block().gradient("positions").values
            - tensor_map_with_grad_3.block().gradient("positions").values
        )
        .pow(2)
        .sum()
    )

    loss_value = loss(output_dict, target_dict)
    torch.testing.assert_close(loss_value, expected_result)


def test_tmap_loss_multiple_datasets_same_target_different_gradients(
    tensor_map_with_grad_1,
    tensor_map_with_grad_1_with_strain,
    tensor_map_with_grad_3_with_strain,
):
    """Test that the loss is computed correctly when two datasets have the same target,
    but different gradients."""

    block = TensorBlock(
        values=torch.empty(0, 1),
        samples=Labels(
            names=["system"],
            values=torch.empty((0, 1), dtype=torch.int32),
        ),
        components=[],
        properties=Labels.range("property", 1),
    )
    block.add_gradient(
        "positions",
        TensorBlock(
            values=torch.empty(0, 1),
            samples=Labels(
                names=["sample"],
                values=torch.empty((0, 1), dtype=torch.int32),
            ),
            components=[],
            properties=Labels.range("property", 1),
        ),
    )
    block.add_gradient(
        "strain",
        TensorBlock(
            values=torch.empty(0, 1),
            samples=Labels(
                names=["sample"],
                values=torch.empty((0, 1), dtype=torch.int32),
            ),
            components=[],
            properties=Labels.range("property", 1),
        ),
    )
    layout = TensorMap(keys=Labels.single(), blocks=[block])

    target_info = TargetInfo(layout=layout, quantity="energy", unit="eV")
    loss_hypers = {
        "output": {
            "type": "mse",
            "weight": 1.0,
            "reduction": "sum",
            "gradients": {
                "positions": {
                    "type": "mse",
                    "weight": 0.5,
                    "reduction": "sum",
                },
                "strain": {
                    "type": "mse",
                    "weight": 0.3,
                    "reduction": "sum",
                },
            },
        },
    }

    loss = LossAggregator(
        targets={"output": target_info},
        config=loss_hypers,
    )

    # Test a case where the target has only one gradient
    output_dict = {
        "output": tensor_map_with_grad_3_with_strain,
    }

    # The target has no `other_gradient`
    target_dict = {
        "output": tensor_map_with_grad_1,
    }

    expected_result = (
        1.0
        * (
            tensor_map_with_grad_1.block().values
            - tensor_map_with_grad_3_with_strain.block().values
        )
        .pow(2)
        .sum()
        + 0.5
        * (
            tensor_map_with_grad_1.block().gradient("positions").values
            - tensor_map_with_grad_3_with_strain.block().gradient("positions").values
        )
        .pow(2)
        .sum()
    )

    loss_value = loss(output_dict, target_dict)
    torch.testing.assert_close(loss_value, expected_result)

    # Test a case where the target has both gradients
    output_dict = {
        "output": tensor_map_with_grad_3_with_strain,
    }

    # The target has no `other_gradient`
    target_dict = {
        "output": tensor_map_with_grad_1_with_strain,
    }

    expected_result = (
        1.0
        * (
            tensor_map_with_grad_1_with_strain.block().values
            - tensor_map_with_grad_3_with_strain.block().values
        )
        .pow(2)
        .sum()
        + 0.5
        * (
            tensor_map_with_grad_1_with_strain.block().gradient("positions").values
            - tensor_map_with_grad_3_with_strain.block().gradient("positions").values
        )
        .pow(2)
        .sum()
        + 0.3
        * (
            tensor_map_with_grad_1_with_strain.block().gradient("strain").values
            - tensor_map_with_grad_3_with_strain.block().gradient("strain").values
        )
        .pow(2)
        .sum()
    )

    loss_value = loss(output_dict, target_dict)
    torch.testing.assert_close(loss_value, expected_result)


# ===== Tests for GaussianCRPSLoss =====


def test_gaussian_crps_loss_known_values():
    """Test GaussianCRPSLoss against known analytical values."""
    # For a standard normal N(0,1) evaluated at x=0:
    # CRPS = sigma * [z(2Phi(z) - 1) + 2phi(z) - 1/sqrt(pi)]
    # where z=0, Phi(0)=0.5, phi(0)=1/sqrt(2*pi)
    # CRPS = 1 * [0 + 2/sqrt(2*pi) - 1/sqrt(pi)] ≈ 0.2337
    loss_fn = GaussianCRPSLoss(reduction="none")

    # Test case 1: mean=0, var=1, target=0
    input_mean = torch.tensor([0.0])
    target = torch.tensor([0.0])
    var = torch.tensor([1.0])

    result = loss_fn(input_mean, target, var)
    expected = 2.0 / math.sqrt(2.0 * math.pi) - 1.0 / math.sqrt(math.pi)
    assert result.item() == pytest.approx(expected, rel=1e-5)

    # Test case 2: When prediction equals target with zero variance
    # CRPS should be small but not exactly zero due to eps
    input_mean = torch.tensor([2.5])
    target = torch.tensor([2.5])
    var = torch.tensor([1e-15])  # Very small variance

    result = loss_fn(input_mean, target, var)
    # With very small variance, z becomes very large, CRPS approaches 0
    assert result.item() < 1e-6


def test_gaussian_crps_loss_reduction_modes():
    """Test that reduction modes work correctly for GaussianCRPSLoss."""
    input_mean = torch.tensor([0.0, 1.0, 2.0])
    target = torch.tensor([0.5, 1.5, 2.5])
    var = torch.tensor([1.0, 1.0, 1.0])

    # Test 'none' reduction
    loss_none = GaussianCRPSLoss(reduction="none")
    result_none = loss_none(input_mean, target, var)
    assert result_none.shape == (3,)

    # Test 'mean' reduction
    loss_mean = GaussianCRPSLoss(reduction="mean")
    result_mean = loss_mean(input_mean, target, var)
    assert result_mean.item() == pytest.approx(result_none.mean().item())

    # Test 'sum' reduction
    loss_sum = GaussianCRPSLoss(reduction="sum")
    result_sum = loss_sum(input_mean, target, var)
    assert result_sum.item() == pytest.approx(result_none.sum().item())


def test_gaussian_crps_loss_invalid_reduction():
    """Test that invalid reduction mode raises error."""
    loss_fn = GaussianCRPSLoss(reduction="invalid")
    input_mean = torch.tensor([0.0])
    target = torch.tensor([0.0])
    var = torch.tensor([1.0])

    with pytest.raises(ValueError, match="invalid is not valid"):
        loss_fn(input_mean, target, var)


def test_gaussian_crps_loss_variance_clamping():
    """Test that variance is clamped to avoid numerical issues."""
    loss_fn = GaussianCRPSLoss(reduction="none", eps=1e-6)

    input_mean = torch.tensor([0.0])
    target = torch.tensor([1.0])
    var = torch.tensor([1e-12])  # Smaller than eps

    # Should not raise error due to clamping
    result = loss_fn(input_mean, target, var)
    assert torch.isfinite(result).all()


@pytest.fixture
def ensemble_tensor_maps():
    """Create tensor maps for ensemble loss testing."""
    # Create a simple ensemble prediction with 3 ensemble members
    # and 2 samples
    n_samples = 2
    n_ensemble = 3
    n_properties = 1

    # Ensemble predictions: shape (n_samples, n_ensemble * n_properties)
    ensemble_values = torch.tensor(
        [
            [1.0, 1.5, 2.0],  # sample 0: ensemble members predict 1.0, 1.5, 2.0
            [3.0, 3.2, 3.1],  # sample 1: ensemble members predict 3.0, 3.2, 3.1
        ]
    )

    # Original prediction (mean): shape (n_samples, n_properties)
    mean_values = torch.tensor([[1.5], [3.1]])

    # Target values: shape (n_samples, n_properties)
    target_values = torch.tensor([[1.6], [3.0]])

    # Create TensorMaps
    samples = Labels.range("sample", n_samples)
    properties_mean = Labels.range("property", n_properties)
    properties_ensemble = Labels.range("property", n_ensemble * n_properties)

    target_block = TensorBlock(
        values=target_values, samples=samples, components=[], properties=properties_mean
    )
    target_map = TensorMap(keys=Labels.single(), blocks=[target_block])

    mean_block = TensorBlock(
        values=mean_values, samples=samples, components=[], properties=properties_mean
    )
    mean_map = TensorMap(keys=Labels.single(), blocks=[mean_block])

    ensemble_block = TensorBlock(
        values=ensemble_values,
        samples=samples,
        components=[],
        properties=properties_ensemble,
    )
    ensemble_map = TensorMap(keys=Labels.single(), blocks=[ensemble_block])

    return {
        "target": target_map,
        "mean": mean_map,
        "ensemble": ensemble_map,
    }


def test_tensormap_gaussian_nll_loss(ensemble_tensor_maps):
    """Test TensorMapGaussianNLLLoss."""
    loss_fn = TensorMapGaussianNLLLoss(
        name="energy",
        gradient=None,
        weight=1.0,
        reduction="mean",
    )

    predictions = {
        "energy": ensemble_tensor_maps["mean"],
        "energy_ensemble": ensemble_tensor_maps["ensemble"],
    }
    targets = {"energy": ensemble_tensor_maps["target"]}

    # Should not raise an error and should return a finite value
    result = loss_fn.compute(predictions, targets)
    assert torch.isfinite(result)


def test_tensormap_gaussian_crps_loss(ensemble_tensor_maps):
    """Test TensorMapGaussianCRPSLoss."""
    loss_fn = TensorMapGaussianCRPSLoss(
        name="energy",
        gradient=None,
        weight=1.0,
        reduction="mean",
    )

    predictions = {
        "energy": ensemble_tensor_maps["mean"],
        "energy_ensemble": ensemble_tensor_maps["ensemble"],
    }
    targets = {"energy": ensemble_tensor_maps["target"]}

    # Should not raise an error
    result = loss_fn.compute(predictions, targets)
    assert torch.isfinite(result)
    assert result.item() >= 0.0  # CRPS should be non-negative
