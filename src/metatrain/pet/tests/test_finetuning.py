import copy
import shutil

import pytest
import torch
from omegaconf import OmegaConf

from metatrain.pet import PET, Trainer
from metatrain.utils.data import Dataset, DatasetInfo
from metatrain.utils.data.readers import read_systems, read_targets
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.finetuning import apply_finetuning_strategy
from metatrain.utils.hypers import init_with_defaults
from metatrain.utils.io import model_from_checkpoint
from metatrain.utils.loss import LossSpecification

from . import DATASET_PATH, DEFAULT_HYPERS, MODEL_HYPERS


def test_lora_finetuning_functionality():
    target_info_dict = {}
    target_info_dict["energy"] = get_energy_target_info(
        "energy", {"quantity": "energy", "unit": "eV"}
    )
    dataset_info = DatasetInfo(
        length_unit="Angstrom", atomic_types=[1, 6, 7, 8], targets=target_info_dict
    )

    model = PET(MODEL_HYPERS, dataset_info)

    finetuning_strategy = {
        "read_from": None,
        "method": "lora",
        "config": {
            "target_modules": ["input_linear", "output_linear"],
            "rank": 4,
            "alpha": 8,
        },
        "inherit_heads": {},
    }

    model = apply_finetuning_strategy(model, finetuning_strategy)
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = sum(p.numel() for p in model.parameters())
    assert num_trainable_params < num_params


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_lora_finetuning_device(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    target_info_dict = {}
    target_info_dict["energy"] = get_energy_target_info(
        "energy", {"quantity": "energy", "unit": "eV"}
    )
    dataset_info = DatasetInfo(
        length_unit="Angstrom", atomic_types=[1, 6, 7, 8], targets=target_info_dict
    )

    model = PET(MODEL_HYPERS, dataset_info).to(device)

    finetuning_strategy = {
        "read_from": None,
        "method": "lora",
        "config": {
            "target_modules": ["input_linear", "output_linear"],
            "rank": 4,
            "alpha": 8,
        },
        "inherit_heads": {},
    }

    model = apply_finetuning_strategy(model, finetuning_strategy)
    for param in model.parameters():
        assert param.device.type == device, f"Parameter {param.name} is not on {device}"


def test_heads_finetuning_functionality():
    target_info_dict = {}
    target_info_dict["energy"] = get_energy_target_info(
        "energy", {"quantity": "energy", "unit": "eV"}
    )
    dataset_info = DatasetInfo(
        length_unit="Angstrom", atomic_types=[1, 6, 7, 8], targets=target_info_dict
    )

    model = PET(MODEL_HYPERS, dataset_info)

    finetuning_strategy = {
        "read_from": None,
        "method": "heads",
        "config": {
            "head_modules": ["node_heads", "edge_heads"],
            "last_layer_modules": ["node_last_layers", "edge_last_layers"],
        },
        "inherit_heads": {},
    }

    model = apply_finetuning_strategy(model, finetuning_strategy)
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = sum(p.numel() for p in model.parameters())
    assert 0 < num_trainable_params < num_params


def test_heads_finetuning_unknown_modules():
    """Unknown 'head_modules'/'last_layer_modules' should raise, not silently
    freeze the whole model."""
    target_info_dict = {}
    target_info_dict["energy"] = get_energy_target_info(
        "energy", {"quantity": "energy", "unit": "eV"}
    )
    dataset_info = DatasetInfo(
        length_unit="Angstrom", atomic_types=[1, 6, 7, 8], targets=target_info_dict
    )

    model = PET(MODEL_HYPERS, dataset_info)

    finetuning_strategy = {
        "read_from": None,
        "method": "heads",
        "config": {
            "head_modules": ["does_not_exist"],
            "last_layer_modules": ["also_does_not_exist"],
        },
        "inherit_heads": {},
    }

    with pytest.raises(ValueError, match="No parameters were found matching"):
        apply_finetuning_strategy(model, finetuning_strategy)


def test_finetuning_restart(monkeypatch, tmp_path):
    """
    Test finetuning with LoRA layers.
    """

    monkeypatch.chdir(tmp_path)
    shutil.copy(DATASET_PATH, "qm9_reduced_100.xyz")

    systems = read_systems(DATASET_PATH)
    systems = [system.to(torch.float32) for system in systems]

    target_info_dict = {}
    target_info_dict["mtt::U0"] = get_energy_target_info(
        "mtt::U0", {"quantity": "energy", "unit": "eV"}
    )

    dataset_info = DatasetInfo(
        length_unit="Angstrom", atomic_types=[1, 6, 7, 8], targets=target_info_dict
    )
    model = PET(MODEL_HYPERS, dataset_info)

    conf = {
        "mtt::U0": {
            "quantity": "energy",
            "read_from": DATASET_PATH,
            "reader": "ase",
            "key": "U0",
            "unit": "eV",
            "type": "scalar",
            "sample_kind": "system",
            "num_subtargets": 1,
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    targets, _ = read_targets(OmegaConf.create(conf))

    # systems in float64 are required for training
    systems = [system.to(torch.float64) for system in systems]

    dataset = Dataset.from_dict({"system": systems, "mtt::U0": targets["mtt::U0"]})

    hypers = copy.deepcopy(DEFAULT_HYPERS)

    hypers["training"]["num_epochs"] = 1

    loss_conf = OmegaConf.create({"mtt::U0": init_with_defaults(LossSpecification)})
    OmegaConf.resolve(loss_conf)
    hypers["training"]["loss"] = loss_conf

    # Pre-training
    trainer = Trainer(hypers["training"])
    trainer.train(
        model=model,
        dtype=torch.float32,
        devices=[torch.device("cpu")],
        train_datasets=[dataset],
        val_datasets=[dataset],
        checkpoint_dir=".",
    )
    trainer.save_checkpoint(model, "tmp.ckpt")

    # Finetuning
    checkpoint = torch.load("tmp.ckpt", weights_only=False, map_location="cpu")
    model_finetune = model_from_checkpoint(checkpoint, context="finetune")
    assert isinstance(model_finetune, PET)
    model_finetune.restart(dataset_info)

    hypers = copy.deepcopy(DEFAULT_HYPERS)

    hypers["training"]["num_epochs"] = 0
    hypers["training"]["loss"] = loss_conf

    hypers["training"]["finetune"] = {
        "read_from": "tmp.ckpt",
        "method": "lora",
        "config": {
            "target_modules": ["input_linear", "output_linear"],
            "rank": 4,
            "alpha": 8,
        },
        "inherit_heads": {},
    }

    trainer = Trainer(hypers["training"])
    trainer.train(
        model=model_finetune,
        dtype=torch.float32,
        devices=[torch.device("cpu")],
        train_datasets=[dataset],
        val_datasets=[dataset],
        checkpoint_dir=".",
    )

    trainer.save_checkpoint(model_finetune, "finetuned.ckpt")

    assert any(["lora_" in name for name, _ in model_finetune.named_parameters()])

    # Finetuning restart
    checkpoint = torch.load("finetuned.ckpt", weights_only=False, map_location="cpu")
    model_finetune_restart = model_from_checkpoint(checkpoint, context="restart")
    assert isinstance(model_finetune_restart, PET)
    model_finetune_restart.restart(dataset_info)

    assert any(
        ["lora_" in name for name, _ in model_finetune_restart.named_parameters()]
    )

    hypers = copy.deepcopy(DEFAULT_HYPERS)

    hypers["training"]["num_epochs"] = 0
    hypers["training"]["loss"] = loss_conf

    hypers["training"]["finetune"] = {
        "read_from": "finetuned.ckpt",
        "method": "heads",
        "config": {
            "head_modules": ["node_heads", "edge_heads"],
            "last_layer_modules": ["node_last_layers", "edge_last_layers"],
        },
        "inherit_heads": {},
    }

    trainer = Trainer(hypers["training"])
    trainer.train(
        model=model_finetune_restart,
        dtype=torch.float32,
        devices=[torch.device("cpu")],
        train_datasets=[dataset],
        val_datasets=[dataset],
        checkpoint_dir=".",
    )


def _two_target_setup():
    """A model pre-trained on ``"energy"``, plus a second, unrelated ``"mtt::U0"``
    target to fine-tune on."""
    old_target = get_energy_target_info("energy", {"quantity": "energy", "unit": "eV"})
    new_target = get_energy_target_info("mtt::U0", {"quantity": "energy", "unit": "eV"})

    old_dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={"energy": old_target},
    )
    model = PET(MODEL_HYPERS, old_dataset_info)

    new_dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={"mtt::U0": new_target},
    )
    return model, new_dataset_info


def _assert_target_absent(model, target_name):
    assert target_name not in model.dataset_info.targets
    assert target_name not in model.supported_outputs()
    assert target_name not in model.backend.node_heads
    assert target_name not in model.backend.edge_heads
    assert target_name not in model.backend.node_last_layers
    assert target_name not in model.backend.edge_last_layers
    for additive_model in model.additive_models:
        assert target_name not in additive_model.outputs
    assert target_name not in model.scaler.outputs


def _assert_target_present(model, target_name):
    assert target_name in model.dataset_info.targets
    assert target_name in model.supported_outputs()
    assert target_name in model.backend.node_heads
    assert target_name in model.backend.edge_heads
    assert target_name in model.backend.node_last_layers
    assert target_name in model.backend.edge_last_layers


def _finetuning_strategy(method, inherit_heads=None):
    if method == "lora":
        config = {
            "target_modules": ["input_linear", "output_linear"],
            "rank": 4,
            "alpha": 8,
        }
    elif method == "heads":
        config = {
            "head_modules": ["node_heads", "edge_heads"],
            "last_layer_modules": ["node_last_layers", "edge_last_layers"],
        }
    else:
        config = {}
    return {
        "read_from": None,
        "method": method,
        "config": config,
        "inherit_heads": inherit_heads or {},
    }


@pytest.mark.parametrize("method", ["full", "lora"])
def test_finetune_full_lora_prunes_stale_targets(method):
    """A target not part of the current full/lora finetuning run's dataset is
    dropped from the model, since its head is no longer compatible with the
    fine-tuned backbone.

    Removal only happens once ``apply_finetuning_strategy`` runs (as it would when
    training actually starts): ``restart`` alone must not remove it yet, since
    ``inherit_heads`` (applied within ``apply_finetuning_strategy``) may still need
    to copy weights from the stale target's head."""
    model, new_dataset_info = _two_target_setup()

    model.restart(new_dataset_info, finetune_method=method)
    _assert_target_present(model, "energy")

    apply_finetuning_strategy(model, _finetuning_strategy(method))

    _assert_target_absent(model, "energy")
    _assert_target_present(model, "mtt::U0")


def test_finetune_full_inherit_heads_then_prunes_source_target():
    """``inherit_heads`` can copy weights from a stale target's head into the new
    target's head; the stale target is only removed afterwards."""
    model, new_dataset_info = _two_target_setup()

    model.restart(new_dataset_info, finetune_method="full")
    apply_finetuning_strategy(
        model, _finetuning_strategy("full", inherit_heads={"mtt::U0": "energy"})
    )

    _assert_target_absent(model, "energy")
    _assert_target_present(model, "mtt::U0")


def test_finetune_heads_keeps_stale_targets():
    """With heads-only finetuning, the backbone is unchanged, so a target not part
    of the current run's dataset must be kept."""
    model, new_dataset_info = _two_target_setup()

    model.restart(new_dataset_info, finetune_method="heads")
    apply_finetuning_strategy(model, _finetuning_strategy("heads"))

    _assert_target_present(model, "energy")
    _assert_target_present(model, "mtt::U0")


def test_restart_without_finetune_method_keeps_stale_targets():
    """A plain restart (not part of a finetuning run) must not prune any target."""
    model, new_dataset_info = _two_target_setup()

    model.restart(new_dataset_info)

    _assert_target_present(model, "energy")
    _assert_target_present(model, "mtt::U0")
