from __future__ import annotations

import copy

from metatensor.torch import TensorBlock, TensorMap
import pytest
import torch

from metatrain.pet.trainer import get_scheduler
from metatrain.utils.architectures import get_default_hypers

from metatrain.experimental.e_pet.model import EPET
from metatrain.experimental.e_pet.trainer import (
    Trainer,
    _AtomicBasisIrrepBalancedLoss,
)

from .test_model import (
    _atomic_basis_sparse_target,
    _atomic_basis_dataset_info,
    _base_model_hypers,
    _build_system,
    _mixed_dataset_info,
)


def _split_lr_training_hypers() -> dict:
    hypers = copy.deepcopy(get_default_hypers("experimental.e_pet")["training"])
    hypers["learning_rate"] = 1.0e-4
    hypers["pet_trunk_learning_rate"] = 2.0e-4
    hypers["tensor_basis_learning_rate"] = 1.0e-3
    hypers["readout_learning_rate"] = 6.0e-4
    return hypers


def _pet_trainer_training_hypers() -> dict:
    hypers = copy.deepcopy(get_default_hypers("experimental.e_pet")["training"])
    hypers["pet_trunk_learning_rate"] = None
    hypers["tensor_basis_learning_rate"] = None
    hypers["readout_learning_rate"] = None
    hypers["coefficient_l2_weight"] = 0.0
    hypers["basis_gram_weight"] = 0.0
    return hypers


def test_custom_training_path_predicate() -> None:
    assert Trainer(_split_lr_training_hypers())._requires_custom_training_path()
    assert not Trainer(_pet_trainer_training_hypers())._requires_custom_training_path()

    hypers = _pet_trainer_training_hypers()
    hypers["scale_property_floor_ratio"] = 1.0e-2
    assert Trainer(hypers)._requires_custom_training_path()

    hypers = _pet_trainer_training_hypers()
    hypers["atomic_basis_irrep_balanced_loss"] = {
        "density": {"weight": 1.0, "scale": "per_irrep_rms"}
    }
    assert Trainer(hypers)._requires_custom_training_path()


def test_split_learning_rate_optimizer_groups_are_disjoint() -> None:
    model = EPET(_base_model_hypers(), _mixed_dataset_info())
    optimizer = Trainer(_split_lr_training_hypers())._build_optimizer(model)

    groups = {group["name"]: group for group in optimizer.param_groups}
    assert set(groups) == {"pet_trunk", "tensor_basis", "readout"}
    assert groups["pet_trunk"]["lr"] == 2.0e-4
    assert groups["tensor_basis"]["lr"] == 1.0e-3
    assert groups["readout"]["lr"] == 6.0e-4

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


def test_spherical_l0_readout_optimizer_group_is_disjoint() -> None:
    hypers = _split_lr_training_hypers()
    hypers["spherical_l0_readout_learning_rate"] = 3.0e-4
    model = EPET(_base_model_hypers(), _atomic_basis_dataset_info())
    optimizer = Trainer(hypers)._build_optimizer(model)

    groups = {group["name"]: group for group in optimizer.param_groups}
    assert set(groups) == {
        "pet_trunk",
        "tensor_basis",
        "readout",
        "spherical_l0_readout",
    }
    assert groups["readout"]["lr"] == 6.0e-4
    assert groups["spherical_l0_readout"]["lr"] == 3.0e-4

    l0_parameter_ids = {
        id(parameter)
        for name, parameter in model.named_parameters()
        if (
            ("node_last_layers." in name or "edge_last_layers." in name)
            and "_o3_lambda_0_o3_sigma_" in name
        )
    }
    grouped_l0_parameter_ids = {
        id(parameter) for parameter in groups["spherical_l0_readout"]["params"]
    }
    assert grouped_l0_parameter_ids == l0_parameter_ids

    readout_parameter_ids = {id(p) for p in groups["readout"]["params"]}
    readout_parameter_names = {
        name
        for name, parameter in model.named_parameters()
        if id(parameter) in readout_parameter_ids
    }
    assert any(name.startswith("node_heads.") for name in readout_parameter_names)
    assert any("_o3_lambda_1_o3_sigma_" in name for name in readout_parameter_names)
    assert not any("_o3_lambda_0_o3_sigma_" in name for name in readout_parameter_names)

    parameter_ids = [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    assert len(parameter_ids) == len(set(parameter_ids))


def test_spherical_l0_dedicated_head_uses_l0_optimizer_group() -> None:
    hypers = _split_lr_training_hypers()
    hypers["spherical_l0_readout_learning_rate"] = 3.0e-4
    model_hypers = _base_model_hypers()
    model_hypers["irrep_head_groups"] = {"density": {"0,1": "scalar"}}
    model = EPET(model_hypers, _atomic_basis_dataset_info())
    optimizer = Trainer(hypers)._build_optimizer(model)

    groups = {group["name"]: group for group in optimizer.param_groups}
    grouped_l0_parameter_ids = {
        id(parameter) for parameter in groups["spherical_l0_readout"]["params"]
    }
    l0_parameter_names = {
        name
        for name, parameter in model.named_parameters()
        if id(parameter) in grouped_l0_parameter_ids
    }

    assert any(name.startswith("node_heads.") for name in l0_parameter_names)
    assert any(name.startswith("edge_heads.") for name in l0_parameter_names)
    assert any("node_last_layers." in name for name in l0_parameter_names)
    assert any("edge_last_layers." in name for name in l0_parameter_names)
    assert all("density_o3_lambda_1_o3_sigma_" not in name for name in l0_parameter_names)
    assert all("density_o3_lambda_2_o3_sigma_" not in name for name in l0_parameter_names)
    assert groups["spherical_l0_readout"]["lr"] == 3.0e-4

    readout_parameter_ids = {id(parameter) for parameter in groups["readout"]["params"]}
    assert grouped_l0_parameter_ids.isdisjoint(readout_parameter_ids)


def test_scale_property_floor_clamps_fitted_scales_and_buffer() -> None:
    hypers = _pet_trainer_training_hypers()
    hypers["scale_property_floor_ratio"] = 1.0e-1
    model = EPET(_base_model_hypers(), _atomic_basis_dataset_info())
    model.scaler.sync_tensor_maps()

    original_scales = model.scaler.model.scales["density"]
    scale_values = iter(
        [
            torch.tensor(1.0e-9, dtype=torch.float64),
            torch.tensor(1.0e-8, dtype=torch.float64),
            torch.tensor(1.0e-7, dtype=torch.float64),
            torch.tensor(1.0e-6, dtype=torch.float64),
            torch.tensor(1.0e-5, dtype=torch.float64),
            torch.tensor(1.0e-4, dtype=torch.float64),
            torch.tensor(1.0e-3, dtype=torch.float64),
        ]
    )
    blocks: list[TensorBlock] = []
    flat_values_before: list[torch.Tensor] = []
    for block in original_scales.blocks():
        values = torch.full_like(block.values, next(scale_values))
        flat_values_before.append(values.reshape(-1))
        blocks.append(
            TensorBlock(
                values=values,
                samples=block.samples,
                components=block.components,
                properties=block.properties,
            )
        )
    model.scaler.model.scales["density"] = TensorMap(original_scales.keys, blocks)

    median_scale = torch.median(torch.cat(flat_values_before))
    expected_floor = median_scale * hypers["scale_property_floor_ratio"]

    Trainer(hypers)._apply_scale_property_floor(model)

    floored_scales = model.scaler.model.scales["density"]
    floored_values = torch.cat(
        [block.values.reshape(-1) for block in floored_scales.blocks()]
    )
    assert torch.all(floored_values >= expected_floor)
    assert torch.any(floored_values == expected_floor)

    model.scaler.sync_tensor_maps()
    reloaded_values = torch.cat(
        [
            block.values.reshape(-1)
            for block in model.scaler.model.scales["density"].blocks()
        ]
    )
    torch.testing.assert_close(reloaded_values, floored_values)


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


def _identity_atomic_basis_reverse_transform(systems, targets, extra):
    return systems, targets, extra


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
        torch.tensor((2 * 2.0**2 + 3 * 20.0**2 + 200.0**2) / 6.0)
        .sqrt()
        .to(torch.float64),
    )
    torch.testing.assert_close(
        loss.group_scales["density"][(1, 1)],
        torch.tensor((3.0**2 + 2 * 30.0**2) / 3.0).sqrt().to(torch.float64),
    )
    torch.testing.assert_close(
        loss.group_scales["density"][(2, 1)],
        torch.tensor((50.0**2 + 500.0**2) / 2.0).sqrt().to(torch.float64),
    )


def test_atomic_basis_irrep_balanced_loss_weights_irreps_equally() -> None:
    model = EPET(_base_model_hypers(), _atomic_basis_dataset_info())
    _set_atomic_basis_scales(model)
    system = _build_system(model)
    target = _atomic_basis_sparse_target(model.dataset_info.targets["density"], system)

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
        _identity_atomic_basis_reverse_transform,
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
    target = _atomic_basis_sparse_target(model.dataset_info.targets["density"], system)

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
        _identity_atomic_basis_reverse_transform,
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
    hypers = _split_lr_training_hypers()
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


def test_custom_trainer_rejects_distributed_training() -> None:
    hypers = _split_lr_training_hypers()
    hypers["distributed"] = True

    with pytest.raises(NotImplementedError, match="distributed training"):
        Trainer(hypers)._validate_custom_training_path(torch.float32)


def test_custom_trainer_rejects_finetuning() -> None:
    hypers = _split_lr_training_hypers()
    hypers["finetune"]["read_from"] = "model.ckpt"

    with pytest.raises(NotImplementedError, match="finetuning"):
        Trainer(hypers)._validate_custom_training_path(torch.float32)
