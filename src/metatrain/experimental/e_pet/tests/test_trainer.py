from __future__ import annotations

import copy

from metatrain.pet.trainer import get_scheduler
from metatrain.utils.architectures import get_default_hypers

from metatrain.experimental.e_pet.model import EPET
from metatrain.experimental.e_pet.trainer import Trainer

from .test_model import _base_model_hypers, _mixed_dataset_info


def _split_lr_training_hypers() -> dict:
    hypers = copy.deepcopy(get_default_hypers("experimental.e_pet")["training"])
    hypers["learning_rate"] = 1.0e-4
    hypers["pet_trunk_learning_rate"] = 2.0e-4
    hypers["tensor_basis_learning_rate"] = 1.0e-3
    hypers["readout_learning_rate"] = 6.0e-4
    return hypers


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
