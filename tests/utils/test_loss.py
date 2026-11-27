# tests/test_losses.py

import math
from pathlib import Path

import pytest
import torch
from metatensor.learn.data import Dataset
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import System
from torch.utils.data import Subset

from metatrain.utils.data import TargetInfo
from metatrain.utils.loss import (
    GaussianCRPSLoss,
    LossAggregator,
    LossType,
    TensorMapEmpiricalCRPSLoss,
    TensorMapGaussianCRPSLoss,
    TensorMapGaussianNLLLoss,
    TensorMapHuberLoss,
    TensorMapMAELoss,
    TensorMapMaskedHuberLoss,
    TensorMapMaskedMAELoss,
    TensorMapMaskedMSELoss,
    TensorMapMSELoss,
    compute_valid_fractions,
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


@pytest.fixture
def spherical_ensemble_tensor_maps():
    """Same as ``ensemble_tensor_maps``, for a spherical target with a lambda=0 and
    a lambda=2 block."""
    n_samples = 2
    n_ensemble = 3
    n_properties = 2

    samples = Labels.range("sample", n_samples)
    properties_mean = Labels.range("property", n_properties)
    properties_ensemble = Labels.range("property", n_ensemble * n_properties)
    keys = Labels(["o3_lambda", "o3_sigma"], torch.tensor([[0, 1], [2, 1]]))

    generator = torch.Generator().manual_seed(0)
    maps = {"target": [], "mean": [], "ensemble": []}
    for o3_lambda in (0, 2):
        components = [Labels.range("o3_mu", 2 * o3_lambda + 1)]
        ensemble_values = torch.randn(
            n_samples, 2 * o3_lambda + 1, n_ensemble, n_properties, generator=generator
        )
        mean_values = ensemble_values.mean(dim=-2)
        values = {
            "target": (mean_values + 0.1, properties_mean),
            "mean": (mean_values, properties_mean),
            "ensemble": (ensemble_values.flatten(-2), properties_ensemble),
        }
        for name, (block_values, properties) in values.items():
            maps[name].append(
                TensorBlock(
                    values=block_values,
                    samples=samples,
                    components=components,
                    properties=properties,
                )
            )
    return {name: TensorMap(keys=keys, blocks=blocks) for name, blocks in maps.items()}


@pytest.mark.parametrize(
    "loss_class",
    [TensorMapGaussianNLLLoss, TensorMapGaussianCRPSLoss, TensorMapEmpiricalCRPSLoss],
)
def test_ensemble_losses_with_components(loss_class, spherical_ensemble_tensor_maps):
    """Ensemble losses must also work for targets that have components and
    multiple blocks."""
    tensor_maps = spherical_ensemble_tensor_maps
    loss_fn = loss_class(
        name="mtt::target",
        gradient=None,
        weight=1.0,
        reduction="mean",
    )

    predictions = {
        "mtt::target": tensor_maps["mean"],
        "mtt::aux::target_ensemble": tensor_maps["ensemble"],
    }
    targets = {"mtt::target": tensor_maps["target"]}

    result = loss_fn.compute(predictions, targets)

    # Each (sample, component) pair of each block must behave as an independent
    # scalar target
    def _as_scalar(tmap):
        values = torch.cat(
            [
                block.values.reshape(-1, block.values.shape[-1])
                for block in tmap.blocks()
            ]
        )
        block = TensorBlock(
            values=values,
            samples=Labels.range("sample", values.shape[0]),
            components=[],
            properties=tmap.block(0).properties,
        )
        return TensorMap(keys=Labels.single(), blocks=[block])

    scalar_predictions = {k: _as_scalar(v) for k, v in predictions.items()}
    scalar_targets = {k: _as_scalar(v) for k, v in targets.items()}
    expected = loss_fn.compute(scalar_predictions, scalar_targets)
    torch.testing.assert_close(result, expected)


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


# ===== Tests for missing (NaN) value normalization =====


def _make_scalar_tmap(
    values: torch.Tensor,
    strain: torch.Tensor | None = None,
    sample_name: str = "sample",
):
    block = TensorBlock(
        values=values,
        samples=Labels(
            names=[sample_name],
            values=torch.arange(values.shape[0], dtype=torch.int32).reshape(-1, 1),
        ),
        components=[],
        properties=Labels("energy", torch.tensor([[0]])),
    )
    if strain is not None:
        block.add_gradient(
            "strain",
            TensorBlock(
                values=strain,
                samples=Labels(
                    names=["sample"],
                    values=torch.arange(strain.shape[0], dtype=torch.int32).reshape(
                        -1, 1
                    ),
                ),
                components=[],
                properties=Labels("energy", torch.tensor([[0]])),
            ),
        )
    return TensorMap(keys=Labels.single(), blocks=[block])


def test_nan_loss_batch_split_invariance():
    """With 1/(B*p) normalization, the average loss over batches is identical whether
    labeled entries are spread evenly across batches or bunched together, and matches
    the mean error over labeled entries."""
    # 8 structures: 2 labeled (p = 0.25), 6 NaN
    p = 0.25
    pred_vals = torch.tensor([[0.0]] * 8)
    targ_vals = torch.tensor(
        [
            [2.0],
            [float("nan")],
            [float("nan")],
            [float("nan")],
            [4.0],
            [float("nan")],
            [float("nan")],
            [float("nan")],
        ]
    )
    # Exact mean squared error over the 2 labeled structures: (4 + 16) / 2 = 10.0
    expected_mean = 10.0

    loss_fn = TensorMapMSELoss(
        name="y", gradient=None, weight=1.0, reduction="mean", valid_fraction=p
    )

    # Split 1: labeled entries spread evenly (1 per batch of 4)
    even_batches = [[0, 1, 2, 3], [4, 5, 6, 7]]
    even_loss = sum(
        loss_fn(
            {"y": _make_scalar_tmap(pred_vals[idx])},
            {"y": _make_scalar_tmap(targ_vals[idx])},
        )
        for idx in even_batches
    ) / len(even_batches)

    # Split 2: labeled entries bunched in the first batch, second batch all-NaN
    bunched_batches = [[0, 4, 1, 2], [3, 5, 6, 7]]
    bunched_loss = sum(
        loss_fn(
            {"y": _make_scalar_tmap(pred_vals[idx])},
            {"y": _make_scalar_tmap(targ_vals[idx])},
        )
        for idx in bunched_batches
    ) / len(bunched_batches)

    assert even_loss.item() == pytest.approx(expected_mean)
    assert bunched_loss.item() == pytest.approx(expected_mean)


@pytest.mark.parametrize("valid_fraction", [None, 0.25, 0.0])
def test_nan_loss_all_nan_batch_zero_finite_grad(valid_fraction):
    """An all-NaN batch produces zero loss with zero, finite gradients."""
    pred_tensor = torch.tensor([[1.0], [2.0], [3.0]], requires_grad=True)
    targ_tensor = torch.full((3, 1), float("nan"))

    loss_fn = TensorMapMSELoss(
        name="y",
        gradient=None,
        weight=1.0,
        reduction="mean",
        valid_fraction=valid_fraction,
    )
    loss = loss_fn(
        {"y": _make_scalar_tmap(pred_tensor)},
        {"y": _make_scalar_tmap(targ_tensor)},
    )
    assert loss.item() == 0.0
    loss.backward()
    assert pred_tensor.grad is not None
    assert torch.isfinite(pred_tensor.grad).all()
    assert torch.equal(pred_tensor.grad, torch.zeros_like(pred_tensor))


def test_nan_loss_sum_reduction_unchanged():
    """reduction='sum' is unaffected by valid_fraction."""
    pred = _make_scalar_tmap(torch.tensor([[1.0], [2.0], [3.0]]))
    targ = _make_scalar_tmap(torch.tensor([[3.0], [float("nan")], [6.0]]))

    loss_default = TensorMapMSELoss(
        name="y", gradient=None, weight=1.0, reduction="sum", valid_fraction=None
    )
    loss_rescaled = TensorMapMSELoss(
        name="y", gradient=None, weight=1.0, reduction="sum", valid_fraction=0.25
    )

    val_default = loss_default({"y": pred}, {"y": targ})
    val_rescaled = loss_rescaled({"y": pred}, {"y": targ})
    # (1-3)^2 + (3-6)^2 = 4 + 9 = 13
    assert val_default.item() == pytest.approx(13.0)
    torch.testing.assert_close(val_rescaled, val_default)


def test_nan_loss_dense_matches_main():
    """With valid_fraction=1.0 and no NaNs, the loss matches batch (main) behavior."""
    pred = _make_scalar_tmap(torch.tensor([[1.0], [2.0], [3.0]]))
    targ = _make_scalar_tmap(torch.tensor([[1.5], [2.5], [4.0]]))

    loss_batch = TensorMapMSELoss(
        name="y", gradient=None, weight=1.0, reduction="mean", valid_fraction=None
    )
    loss_dense = TensorMapMSELoss(
        name="y", gradient=None, weight=1.0, reduction="mean", valid_fraction=1.0
    )

    torch.testing.assert_close(
        loss_dense({"y": pred}, {"y": targ}),
        loss_batch({"y": pred}, {"y": targ}),
    )


def test_nan_loss_sparse_clamping():
    """Very small valid_fraction is clamped continuously to min_valid_fraction."""
    pred = _make_scalar_tmap(torch.tensor([[0.0], [0.0]]))
    targ = _make_scalar_tmap(torch.tensor([[2.0], [float("nan")]]))

    # Raw 1/B MSE = (4 + 0) / 2 = 2.0
    loss_at_cutoff = TensorMapMSELoss(
        name="y",
        gradient=None,
        weight=1.0,
        reduction="mean",
        valid_fraction=0.05,
        min_valid_fraction=0.05,
    )
    loss_below_cutoff = TensorMapMSELoss(
        name="y",
        gradient=None,
        weight=1.0,
        reduction="mean",
        valid_fraction=0.01,
        min_valid_fraction=0.05,
    )

    assert loss_at_cutoff({"y": pred}, {"y": targ}).item() == pytest.approx(2.0 / 0.05)
    torch.testing.assert_close(
        loss_below_cutoff({"y": pred}, {"y": targ}),
        loss_at_cutoff({"y": pred}, {"y": targ}),
    )


def test_masked_loss_all_false_mask_returns_zero():
    """len(all_targets_flattened) == 0 is still reachable via an all-False mask."""
    pred = _make_scalar_tmap(torch.tensor([[1.0], [2.0]]))
    targ = _make_scalar_tmap(torch.tensor([[3.0], [4.0]]))
    mask = _make_scalar_tmap(torch.tensor([[False], [False]]))

    loss_fn = TensorMapMaskedMSELoss(
        name="y", gradient=None, weight=1.0, reduction="mean", valid_fraction=0.5
    )
    result = loss_fn({"y": pred}, {"y": targ}, {"y_mask": mask})
    assert result.item() == 0.0


def test_loss_aggregator_missing_values_modes_and_validation():
    """LossAggregator supports dataset_fraction, batch, clamping, fallback, and
    non-pointwise losses."""
    layout = _make_scalar_tmap(
        torch.empty(0, 1), strain=torch.empty(0, 1), sample_name="system"
    )
    target_info = TargetInfo(layout=layout, quantity="energy", unit="eV")

    pred = _make_scalar_tmap(
        torch.tensor([[0.0], [0.0]]), strain=torch.tensor([[0.0], [0.0]])
    )
    targ = _make_scalar_tmap(
        torch.tensor([[1.0], [1.0]]), strain=torch.tensor([[2.0], [float("nan")]])
    )
    valid_fractions = {("energy", None): 1.0, ("energy", "strain"): 0.25}

    # dataset_fraction mode on strain: energy MSE = 1.0, strain MSE = (4/2) / 0.25 = 8.0
    agg_df = LossAggregator(
        targets={"energy": target_info},
        config={
            "energy": {
                "type": "mse",
                "weight": 1.0,
                "reduction": "mean",
                "missing_values": "dataset_fraction",
                "min_valid_fraction": 0.05,
                "gradients": {
                    "strain": {
                        "type": "mse",
                        "weight": 1.0,
                        "reduction": "mean",
                        "missing_values": "dataset_fraction",
                        "min_valid_fraction": 0.05,
                    }
                },
            }
        },
        valid_fractions=valid_fractions,
    )
    assert agg_df({"energy": pred}, {"energy": targ}).item() == pytest.approx(9.0)

    # batch mode on strain: energy MSE = 1.0, strain MSE = 4/2 = 2.0
    agg_batch = LossAggregator(
        targets={"energy": target_info},
        config={
            "energy": {
                "type": "mse",
                "weight": 1.0,
                "reduction": "mean",
                "gradients": {
                    "strain": {
                        "type": "mse",
                        "weight": 1.0,
                        "reduction": "mean",
                        "missing_values": "batch",
                    }
                },
            }
        },
        valid_fractions=valid_fractions,
    )
    assert agg_batch({"energy": pred}, {"energy": targ}).item() == pytest.approx(3.0)

    # Fallback to 1/B when valid_fractions is not provided
    agg_fallback = LossAggregator(
        targets={"energy": target_info},
        config={
            "energy": {
                "type": "mse",
                "weight": 1.0,
                "reduction": "mean",
                "gradients": {
                    "strain": {
                        "type": "mse",
                        "weight": 1.0,
                        "reduction": "mean",
                    }
                },
            }
        },
    )
    assert agg_fallback({"energy": pred}, {"energy": targ}).item() == pytest.approx(3.0)

    # Non-pointwise loss accepts default missing_values/min_valid_fraction config
    agg_ens = LossAggregator(
        targets={"energy": target_info},
        config={
            "energy": {
                "type": "gaussian_nll_ensemble",
                "weight": 1.0,
                "reduction": "mean",
                "missing_values": "dataset_fraction",
                "min_valid_fraction": 0.05,
                "gradients": {},
            }
        },
        valid_fractions=valid_fractions,
    )
    assert isinstance(agg_ens.losses["energy"], TensorMapGaussianNLLLoss)

    # Validation errors
    with pytest.raises(ValueError, match="Unknown missing_values"):
        LossAggregator(
            targets={"energy": target_info},
            config={
                "energy": {
                    "type": "mse",
                    "weight": 1.0,
                    "reduction": "mean",
                    "missing_values": "invalid",  # type: ignore[typeddict-item]
                    "gradients": {},
                }
            },
        )
    with pytest.raises(ValueError, match="min_valid_fraction"):
        LossAggregator(
            targets={"energy": target_info},
            config={
                "energy": {
                    "type": "mse",
                    "weight": 1.0,
                    "reduction": "mean",
                    "min_valid_fraction": 0.0,
                    "gradients": {},
                }
            },
        )
    with pytest.raises(ValueError, match="valid_fraction"):
        TensorMapMSELoss(
            name="y", gradient=None, weight=1.0, reduction="mean", valid_fraction=1.5
        )


def test_compute_valid_fractions_values_gradients_and_transforms():
    """compute_valid_fractions counts entries for values and gradients across datasets
    and respects optional collate transforms (e.g. NaN padding)."""

    def _make_system(n_atoms: int) -> System:
        return System(
            types=torch.ones(n_atoms, dtype=torch.int32),
            positions=torch.zeros((n_atoms, 3), dtype=torch.float64),
            cell=torch.eye(3, dtype=torch.float64),
            pbc=torch.tensor([True, True, True]),
        )

    def _make_energy(i: int, n_atoms: int, valid_forces: bool, valid_stress: bool):
        block = TensorBlock(
            values=torch.tensor([[float(i)]], dtype=torch.float64),
            samples=Labels("system", torch.tensor([[i]])),
            components=[],
            properties=Labels("energy", torch.tensor([[0]])),
        )
        f_vals = (
            torch.zeros((n_atoms, 3, 1), dtype=torch.float64)
            if valid_forces
            else torch.full((n_atoms, 3, 1), float("nan"), dtype=torch.float64)
        )
        block.add_gradient(
            "positions",
            TensorBlock(
                values=f_vals,
                samples=Labels(
                    ["sample", "atom"],
                    torch.tensor([[0, a] for a in range(n_atoms)]),
                ),
                components=[Labels.range("xyz", 3)],
                properties=Labels("energy", torch.tensor([[0]])),
            ),
        )
        s_vals = (
            torch.zeros((1, 3, 3, 1), dtype=torch.float64)
            if valid_stress
            else torch.full((1, 3, 3, 1), float("nan"), dtype=torch.float64)
        )
        block.add_gradient(
            "strain",
            TensorBlock(
                values=s_vals,
                samples=Labels("sample", torch.tensor([[0]])),
                components=[Labels.range("xyz_1", 3), Labels.range("xyz_2", 3)],
                properties=Labels("energy", torch.tensor([[0]])),
            ),
        )
        return TensorMap(keys=Labels.single(), blocks=[block])

    # Structure 0: 1 atom, valid forces (3 entries), valid stress (9 entries)
    # Structure 1: 3 atoms, NaN forces (9 entries), NaN stress (9 entries)
    systems = [_make_system(1), _make_system(3)]
    energies = [
        _make_energy(0, 1, valid_forces=True, valid_stress=True),
        _make_energy(1, 3, valid_forces=False, valid_stress=False),
    ]
    dataset = Dataset(system=systems, energy=energies)
    layout_block = TensorBlock(
        values=torch.empty((0, 1), dtype=torch.float64),
        samples=Labels("system", torch.empty((0, 1), dtype=torch.int32)),
        components=[],
        properties=Labels("energy", torch.tensor([[0]])),
    )
    layout_block.add_gradient(
        "positions",
        TensorBlock(
            values=torch.empty((0, 3, 1), dtype=torch.float64),
            samples=Labels(["sample", "atom"], torch.empty((0, 2), dtype=torch.int32)),
            components=[Labels.range("xyz", 3)],
            properties=Labels("energy", torch.tensor([[0]])),
        ),
    )
    layout_block.add_gradient(
        "strain",
        TensorBlock(
            values=torch.empty((0, 3, 3, 1), dtype=torch.float64),
            samples=Labels("sample", torch.empty((0, 1), dtype=torch.int32)),
            components=[Labels.range("xyz_1", 3), Labels.range("xyz_2", 3)],
            properties=Labels("energy", torch.tensor([[0]])),
        ),
    )
    target_info = TargetInfo(
        layout=TensorMap(keys=Labels.single(), blocks=[layout_block]),
        quantity="energy",
        unit="eV",
    )
    targets = {"energy": target_info}

    # Direct (raw dataset, no transforms)
    fracs = compute_valid_fractions([dataset], targets)
    assert fracs[("energy", None)] == pytest.approx(1.0)
    # Forces weighted by atom entries: 3 valid out of 3 + 9 = 12 -> 0.25
    assert fracs[("energy", "positions")] == pytest.approx(3 / 12)
    # Stress: 9 valid out of 18 -> 0.5
    assert fracs[("energy", "strain")] == pytest.approx(0.5)

    # Subset support (only structure 0 -> all 1.0)
    subset_fracs = compute_valid_fractions([Subset(dataset, [0])], targets)
    assert subset_fracs[("energy", "strain")] == pytest.approx(1.0)

    # Transform that NaN-pads energy with an extra column -> valid fraction becomes 0.5
    def nan_pad_transform(sys_list, targ_dict, extra_dict):
        tmap = targ_dict["energy"]
        b = tmap.block(0)
        padded = torch.cat([b.values, torch.full_like(b.values, float("nan"))], dim=-1)
        new_block = TensorBlock(
            values=padded,
            samples=b.samples,
            components=b.components,
            properties=Labels.range("energy", 2),
        )
        targ_dict = dict(targ_dict)
        targ_dict["energy"] = TensorMap(keys=tmap.keys, blocks=[new_block])
        return sys_list, targ_dict, extra_dict

    transformed_fracs = compute_valid_fractions(
        [dataset], targets, transforms=[nan_pad_transform]
    )
    assert transformed_fracs[("energy", None)] == pytest.approx(0.5)

    # Skips scan when config doesn't require dataset_fraction on any term
    skipped = compute_valid_fractions(
        [dataset],
        targets,
        config={
            "energy": {
                "type": "mse",
                "weight": 1.0,
                "reduction": "mean",
                "missing_values": "batch",
                "gradients": {
                    "positions": {
                        "type": "mse",
                        "weight": 1.0,
                        "reduction": "sum",
                    },
                    "strain": {
                        "type": "mse",
                        "weight": 1.0,
                        "reduction": "mean",
                        "missing_values": "batch",
                    },
                },
            }
        },
    )
    assert skipped == {}
