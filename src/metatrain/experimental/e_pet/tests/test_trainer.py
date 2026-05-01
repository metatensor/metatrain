from __future__ import annotations

import copy

import pytest
import torch

from metatrain.pet.trainer import get_scheduler
from metatrain.utils.architectures import get_default_hypers

from metatrain.experimental.e_pet.model import EPET
from metatrain.experimental.e_pet.trainer import Trainer

from .test_model import (
    _atomic_basis_dataset_info,
    _base_model_hypers,
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
