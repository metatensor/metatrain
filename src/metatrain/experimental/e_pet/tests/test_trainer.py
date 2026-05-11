from __future__ import annotations

import copy

import pytest
import torch
from metatensor.torch import TensorBlock, TensorMap

from metatrain.experimental.e_pet.model import EPET
from metatrain.experimental.e_pet.trainer import (
    Trainer,
    _AtomicBasisIrrepBalancedLoss,
)
from metatrain.pet.trainer import _apply_finetuning_if_requested, get_scheduler
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data.atomic_basis_helpers import (
    get_prepare_atomic_basis_targets_transform,
)

from .test_model import (
    _atomic_basis_dataset_info,
    _atomic_basis_sparse_target,
    _base_model_hypers,
    _build_system,
    _mixed_dataset_info,
    _scalar_dataset_info,
    _system_index_extra,
)


def _basis_lr_training_hypers() -> dict:
    hypers = copy.deepcopy(get_default_hypers("experimental.e_pet")["training"])
    hypers["learning_rate"] = 1.0e-4
    hypers["tensor_basis_learning_rate"] = 1.0e-3
    return hypers


def _pet_trainer_training_hypers() -> dict:
    hypers = copy.deepcopy(get_default_hypers("experimental.e_pet")["training"])
    hypers["tensor_basis_learning_rate"] = None
    hypers["coefficient_l2_weight"] = 0.0
    hypers["basis_gram_weight"] = 0.0
    return hypers


def test_custom_training_path_predicate() -> None:
    assert Trainer(_basis_lr_training_hypers())._requires_custom_training_path()
    assert not Trainer(_pet_trainer_training_hypers())._requires_custom_training_path()

    hypers = _pet_trainer_training_hypers()
    hypers["tensor_basis_learning_rate"] = hypers["learning_rate"]
    assert not Trainer(hypers)._requires_custom_training_path()

    hypers = _pet_trainer_training_hypers()
    hypers["atomic_basis_irrep_balanced_loss"] = {
        "density": {"weight": 1.0, "scale": "per_irrep_rms"}
    }
    assert Trainer(hypers)._requires_custom_training_path()

    scalar_only_model = EPET(_base_model_hypers(), _scalar_dataset_info())
    assert not Trainer(_basis_lr_training_hypers())._requires_custom_training_path(
        scalar_only_model
    )

    spherical_model = EPET(_base_model_hypers(), _mixed_dataset_info())
    assert Trainer(_basis_lr_training_hypers())._requires_custom_training_path(
        spherical_model
    )


def test_tensor_basis_learning_rate_optimizer_groups_are_disjoint() -> None:
    model = EPET(_base_model_hypers(), _atomic_basis_dataset_info())
    optimizer = Trainer(_basis_lr_training_hypers())._build_optimizer(model)

    groups = {group["name"]: group for group in optimizer.param_groups}
    assert set(groups) == {"pet_and_readout", "tensor_basis"}
    assert groups["pet_and_readout"]["lr"] == 1.0e-4
    assert groups["tensor_basis"]["lr"] == 1.0e-3

    parameter_ids = [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    assert len(parameter_ids) == len(set(parameter_ids))
    for group in optimizer.param_groups:
        assert sum(parameter.numel() for parameter in group["params"]) > 0

    expected_trainable_ids = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    assert set(parameter_ids) == expected_trainable_ids

    pet_and_readout_parameter_ids = {
        id(parameter) for parameter in groups["pet_and_readout"]["params"]
    }
    pet_and_readout_parameter_names = {
        name
        for name, parameter in model.named_parameters()
        if id(parameter) in pet_and_readout_parameter_ids
    }
    assert any(
        name.startswith("node_heads.") for name in pet_and_readout_parameter_names
    )
    assert any(
        name.startswith("edge_heads.") for name in pet_and_readout_parameter_names
    )
    assert any(
        "_o3_lambda_0_o3_sigma_" in name
        for name in pet_and_readout_parameter_names
    )
    assert any(
        "_o3_lambda_1_o3_sigma_" in name
        for name in pet_and_readout_parameter_names
    )
    assert all(
        not name.startswith("basis_calculators.")
        for name in pet_and_readout_parameter_names
    )


def test_equal_tensor_basis_learning_rate_uses_single_optimizer_group() -> None:
    hypers = _pet_trainer_training_hypers()
    hypers["tensor_basis_learning_rate"] = hypers["learning_rate"]
    model = EPET(_base_model_hypers(), _mixed_dataset_info())

    optimizer = Trainer(hypers)._build_optimizer(model)

    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["lr"] == hypers["learning_rate"]
    parameter_ids = {id(parameter) for parameter in optimizer.param_groups[0]["params"]}
    expected_parameter_ids = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    assert parameter_ids == expected_parameter_ids


def _set_atomic_basis_scales(model: EPET, variable_by_species: bool = False) -> None:
    model.scaler.sync_tensor_maps()
    scales = model.scaler.model.scales["density"]
    blocks = []
    for key, block in scales.items():
        values = torch.ones_like(block.values)
        if variable_by_species and int(key["o3_lambda"]) == 0:
            values[:] = torch.tensor([[2.0], [20.0], [200.0]], dtype=values.dtype)
        elif variable_by_species and int(key["o3_lambda"]) == 1:
            values[:] = torch.tensor([[3.0], [30.0], [300.0]], dtype=values.dtype)
        elif variable_by_species:
            values[:] = torch.tensor([[5.0], [50.0], [500.0]], dtype=values.dtype)
        elif int(key["o3_lambda"]) == 0:
            values[:] = 2.0
        elif int(key["o3_lambda"]) == 1:
            values[:] = 3.0
        else:
            values[:] = 5.0
        blocks.append(
            TensorBlock(
                values=values,
                samples=block.samples,
                components=block.components,
                properties=block.properties,
            )
        )
    model.scaler.model.scales["density"] = TensorMap(scales.keys, blocks)


def _dense_scaled_atomic_basis_target(model: EPET, system):
    sparse_target = _atomic_basis_sparse_target(
        model.dataset_info.targets["density"], system
    )
    prepare_atomic_basis_targets, reverse_atomic_basis_transform = (
        get_prepare_atomic_basis_targets_transform(model.dataset_info.targets, {})
    )
    _, dense_targets, _ = prepare_atomic_basis_targets(
        [system],
        {"density": sparse_target},
        _system_index_extra(),
    )
    scaled_targets = model.scaler([system], dense_targets, remove=True)
    return scaled_targets["density"], reverse_atomic_basis_transform


def test_atomic_basis_irrep_balanced_loss_uses_one_scale_per_irrep() -> None:
    model = EPET(_base_model_hypers(), _atomic_basis_dataset_info())
    _set_atomic_basis_scales(model, variable_by_species=True)

    loss = _AtomicBasisIrrepBalancedLoss(
        target_infos=model.dataset_info.targets,
        config={"density": {"weight": 1.0, "scale": "per_irrep_rms"}},
        scaler=model.scaler,
        scale_targets=True,
    )

    torch.testing.assert_close(
        loss.group_scales["density"][(0, 1)],
        torch.tensor((3 * 2.0**2 + 3 * 20.0**2 + 3 * 200.0**2) / 9.0)
        .sqrt()
        .to(torch.float64),
    )
    torch.testing.assert_close(
        loss.group_scales["density"][(1, 1)],
        torch.tensor((2 * 3.0**2 + 2 * 30.0**2 + 2 * 300.0**2) / 6.0)
        .sqrt()
        .to(torch.float64),
    )
    torch.testing.assert_close(
        loss.group_scales["density"][(2, 1)],
        torch.tensor((5.0**2 + 50.0**2 + 500.0**2) / 3.0)
        .sqrt()
        .to(torch.float64),
    )


def test_atomic_basis_irrep_balanced_loss_keeps_zero_scale_irrep() -> None:
    model = EPET(_base_model_hypers(), _atomic_basis_dataset_info())
    _set_atomic_basis_scales(model)

    scales = model.scaler.model.scales["density"]
    blocks = []
    zero_scale_dtype = scales.block(0).values.dtype
    for key, block in scales.items():
        values = block.values.clone()
        if int(key["o3_lambda"]) == 2:
            values.zero_()
            zero_scale_dtype = values.dtype
        blocks.append(
            TensorBlock(
                values=values,
                samples=block.samples,
                components=block.components,
                properties=block.properties,
            )
        )
    model.scaler.model.scales["density"] = TensorMap(scales.keys, blocks)

    loss = _AtomicBasisIrrepBalancedLoss(
        target_infos=model.dataset_info.targets,
        config={"density": {"weight": 1.0, "scale": "per_irrep_rms"}},
        scaler=model.scaler,
        scale_targets=True,
    )

    assert (2, 1) in loss.group_scales["density"]
    torch.testing.assert_close(
        loss.group_scales["density"][(2, 1)],
        torch.tensor(torch.finfo(zero_scale_dtype).eps, dtype=zero_scale_dtype),
    )


def test_atomic_basis_irrep_balanced_loss_weights_irreps_equally() -> None:
    model = EPET(_base_model_hypers(), _atomic_basis_dataset_info())
    _set_atomic_basis_scales(model)
    system = _build_system(model)
    target, reverse_atomic_basis_transform = _dense_scaled_atomic_basis_target(
        model, system
    )

    prediction_blocks = []
    for key, target_block in target.items():
        values = target_block.values.clone()
        values += 1.0 if int(key["o3_lambda"]) == 0 else 2.0
        prediction_blocks.append(
            TensorBlock(
                values=values,
                samples=target_block.samples,
                components=target_block.components,
                properties=target_block.properties,
            )
        )
    prediction = TensorMap(target.keys, prediction_blocks)

    loss = _AtomicBasisIrrepBalancedLoss(
        target_infos=model.dataset_info.targets,
        config={"density": {"weight": 1.0, "scale": "per_irrep_rms"}},
        scaler=model.scaler,
        scale_targets=True,
    )

    value = loss.compute(
        [system],
        {"density": prediction},
        {"density": target},
        reverse_atomic_basis_transform,
        model.scaler,
    )

    # In scaled training space the residuals are 1 for l=0 and 2 for l=1/l=2.
    # The helper restores physical units, then divides by the same per-irrep RMS
    # scale. Each irrep therefore contributes 1, 4 and 4, averaged equally.
    torch.testing.assert_close(value, torch.tensor(3.0))


def test_atomic_basis_irrep_balanced_loss_ignores_nan_targets() -> None:
    model = EPET(_base_model_hypers(), _atomic_basis_dataset_info())
    _set_atomic_basis_scales(model)
    system = _build_system(model)
    target, reverse_atomic_basis_transform = _dense_scaled_atomic_basis_target(
        model, system
    )

    target_blocks = []
    prediction_blocks = []
    for key, block in target.items():
        target_values = block.values.clone()
        prediction_values = block.values.clone()
        if int(key["o3_lambda"]) == 0:
            target_values.reshape(-1)[0] = torch.nan
            prediction_values += 1.0
        target_blocks.append(
            TensorBlock(
                values=target_values,
                samples=block.samples,
                components=block.components,
                properties=block.properties,
            )
        )
        prediction_blocks.append(
            TensorBlock(
                values=prediction_values,
                samples=block.samples,
                components=block.components,
                properties=block.properties,
            )
        )

    loss = _AtomicBasisIrrepBalancedLoss(
        target_infos=model.dataset_info.targets,
        config={"density": {"weight": 1.0, "scale": "per_irrep_rms"}},
        scaler=model.scaler,
        scale_targets=True,
    )
    value = loss.compute(
        [system],
        {"density": TensorMap(target.keys, prediction_blocks)},
        {"density": TensorMap(target.keys, target_blocks)},
        reverse_atomic_basis_transform,
        model.scaler,
    )

    torch.testing.assert_close(value, torch.tensor(1.0 / 3.0))


def test_atomic_basis_irrep_balanced_loss_rejects_invalid_targets() -> None:
    model = EPET(_base_model_hypers(), _mixed_dataset_info())
    model.scaler.sync_tensor_maps()

    with pytest.raises(ValueError, match="spherical atomic-basis"):
        _AtomicBasisIrrepBalancedLoss(
            target_infos=model.dataset_info.targets,
            config={"quadrupole": {"weight": 1.0, "scale": "per_irrep_rms"}},
            scaler=model.scaler,
            scale_targets=True,
        )

    with pytest.raises(ValueError, match="scale_targets=true"):
        _AtomicBasisIrrepBalancedLoss(
            target_infos=model.dataset_info.targets,
            config={"quadrupole": {"weight": 1.0, "scale": "per_irrep_rms"}},
            scaler=model.scaler,
            scale_targets=False,
        )


def test_custom_trainer_restores_restart_optimizer_and_scheduler_state() -> None:
    hypers = _basis_lr_training_hypers()
    model = EPET(_base_model_hypers(), _mixed_dataset_info())
    trainer = Trainer(hypers)

    optimizer = trainer._build_optimizer(model)
    lr_scheduler = get_scheduler(optimizer, hypers, steps_per_epoch=3)

    optimizer_state = copy.deepcopy(optimizer.state_dict())
    optimizer_state["param_groups"][0]["lr"] = 9.0e-4
    scheduler_state = copy.deepcopy(lr_scheduler.state_dict())
    scheduler_state["last_epoch"] = 5

    trainer.optimizer_state_dict = optimizer_state
    trainer.scheduler_state_dict = scheduler_state

    restored_optimizer = trainer._build_optimizer(model)
    restored_scheduler = get_scheduler(restored_optimizer, hypers, steps_per_epoch=3)
    trainer._load_restart_state(model, restored_optimizer, restored_scheduler)

    assert restored_optimizer.param_groups[0]["lr"] == 9.0e-4
    assert restored_scheduler.state_dict()["last_epoch"] == 5


def test_custom_trainer_accepts_distributed_fixed_batch_training() -> None:
    hypers = _basis_lr_training_hypers()
    hypers["distributed"] = True

    Trainer(hypers)._validate_custom_training_path(torch.float32)


def test_custom_trainer_supports_heads_finetuning() -> None:
    hypers = _basis_lr_training_hypers()
    hypers["finetune"] = {
        "read_from": "model.ckpt",
        "method": "heads",
        "config": {
            "head_modules": ["node_heads", "edge_heads"],
            "last_layer_modules": ["node_last_layers", "edge_last_layers"],
        },
        "inherit_heads": {},
    }
    trainer = Trainer(hypers)
    trainer._validate_custom_training_path(torch.float32)

    model = _apply_finetuning_if_requested(
        EPET(_base_model_hypers(), _mixed_dataset_info()),
        hypers,
    )
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }

    assert model.finetune_config == hypers["finetune"]
    assert trainable_names
    assert all(
        name.startswith(
            (
                "node_heads.",
                "edge_heads.",
                "node_last_layers.",
                "edge_last_layers.",
            )
        )
        for name in trainable_names
    )

    optimizer = trainer._build_optimizer(model)
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["lr"] == hypers["learning_rate"]


def test_custom_trainer_rejects_max_atoms_per_batch() -> None:
    hypers = _basis_lr_training_hypers()
    hypers["max_atoms_per_batch"] = 128

    with pytest.raises(NotImplementedError, match="max_atoms_per_batch"):
        Trainer(hypers)._validate_custom_training_path(torch.float32)
