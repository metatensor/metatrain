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
    TensorMapGaussianCRPSLoss,
    TensorMapGaussianNLLLoss,
    TensorMapHuberLoss,
    TensorMapMAELoss,
    TensorMapMaskedHuberLoss,
    TensorMapMaskedMAELoss,
    TensorMapMaskedMSELoss,
    TensorMapMSELoss,
    create_loss,
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
