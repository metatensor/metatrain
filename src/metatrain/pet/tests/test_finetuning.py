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
from metatrain.utils.io import model_from_checkpoint, trainer_from_checkpoint
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


def _two_coexisting_targets_model():
    """A single model with two co-existing energy-like targets, ``"energy"`` and
    ``"energy_new"``, used to test ``PET.set_default_target``."""
    target_a = get_energy_target_info("energy", {"quantity": "energy", "unit": "eV"})
    target_b = get_energy_target_info(
        "energy_new", {"quantity": "energy", "unit": "eV"}
    )
    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={"energy": target_a, "energy_new": target_b},
    )
    return PET(MODEL_HYPERS, dataset_info)


def _target_params(model, target_name):
    return {
        name: param
        for name, param in model.named_parameters()
        if f".{target_name}." in name
    }


def _fill_target_params(model, target_name, value):
    for param in _target_params(model, target_name).values():
        param.data.fill_(value)


def test_set_default_target_copies_head_weights():
    """``set_default_target`` overwrites the destination's head/last-layer weights
    with the source's, and the two remain independent afterwards."""
    model = _two_coexisting_targets_model()
    _fill_target_params(model, "energy", 1.0)
    _fill_target_params(model, "energy_new", 2.0)

    model.set_default_target("energy_new")

    energy_params = _target_params(model, "energy")
    energy_new_params = _target_params(model, "energy_new")
    assert len(energy_params) == len(energy_new_params) > 0
    for name, param in energy_new_params.items():
        dest_name = name.replace("energy_new", "energy")
        torch.testing.assert_close(param, energy_params[dest_name])

    # Independence: mutating the source afterwards must not affect the copy.
    _fill_target_params(model, "energy_new", 3.0)
    for param in energy_params.values():
        assert not torch.allclose(param, torch.full_like(param, 3.0))


def test_set_default_target_copies_composition_and_scaler_state():
    """``set_default_target`` also copies composition/scaler per-target state
    (not reachable via ``named_parameters()``), and it is independent of the
    source afterwards."""
    model = _two_coexisting_targets_model()
    composition_model = model.additive_models[0]

    model.set_default_target("energy_new")

    assert "energy" in composition_model.outputs
    assert "energy" in model.scaler.outputs

    source_buffer = composition_model.__getattr__("energy_new_composition_buffer")
    dest_buffer = composition_model.__getattr__("energy_composition_buffer")
    assert torch.equal(source_buffer, dest_buffer)
    assert source_buffer is not dest_buffer


def test_set_default_target_overwrites_existing_destination():
    """If the destination target already exists, it is fully replaced."""
    model = _two_coexisting_targets_model()
    _fill_target_params(model, "energy", 1.0)
    _fill_target_params(model, "energy_new", 2.0)

    model.set_default_target("energy_new")

    for param in _target_params(model, "energy").values():
        assert torch.allclose(param, torch.full_like(param, 2.0))


def test_set_default_target_unknown_source_raises():
    model = _two_coexisting_targets_model()

    with pytest.raises(ValueError, match="is not a target of this model"):
        model.set_default_target("does_not_exist")


def test_finetuning_restart_does_not_reapply_inherit_heads(monkeypatch, tmp_path):
    """Restarting an interrupted finetuning run must not redo the one-time
    ``inherit_heads`` weight copy: the destination head's weights, once trained,
    should survive a restart with ``num_epochs=0`` unchanged."""
    monkeypatch.chdir(tmp_path)
    shutil.copy(DATASET_PATH, "qm9_reduced_100.xyz")

    systems = [system.to(torch.float64) for system in read_systems(DATASET_PATH)]

    old_target = get_energy_target_info("energy", {"quantity": "energy", "unit": "eV"})
    old_dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={"energy": old_target},
    )
    model = PET(MODEL_HYPERS, old_dataset_info)

    energy_conf = {
        "energy": {
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
    energy_targets, _ = read_targets(OmegaConf.create(energy_conf))
    energy_dataset = Dataset.from_dict(
        {"system": systems, "energy": energy_targets["energy"]}
    )

    loss_conf = OmegaConf.create({"energy": init_with_defaults(LossSpecification)})
    OmegaConf.resolve(loss_conf)

    hypers = copy.deepcopy(DEFAULT_HYPERS)
    hypers["training"]["num_epochs"] = 1
    hypers["training"]["loss"] = loss_conf

    # Pre-training on "energy".
    trainer = Trainer(hypers["training"])
    trainer.train(
        model=model,
        dtype=torch.float32,
        devices=[torch.device("cpu")],
        train_datasets=[energy_dataset],
        val_datasets=[energy_dataset],
        checkpoint_dir=".",
    )
    trainer.save_checkpoint(model, "pretrained.ckpt")

    # Finetune: introduce a new target "mtt::U0", inheriting its initial weights
    # from "energy". Use ``heads`` so "energy" is not pruned as stale, and the new
    # target's head is trainable (and therefore actually changes during training).
    # The finetuning run's dataset only trains on "mtt::U0" (matching the flow
    # where "energy" is not part of the current run's training set) --
    # ``inherit_heads`` copies weights directly and does not require the source
    # target to be part of the current training batch.
    new_target = get_energy_target_info("mtt::U0", {"quantity": "energy", "unit": "eV"})
    new_dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={"mtt::U0": new_target},
    )
    conf = {"mtt::U0": dict(energy_conf["energy"])}
    targets, _ = read_targets(OmegaConf.create(conf))
    dataset = Dataset.from_dict({"system": systems, "mtt::U0": targets["mtt::U0"]})

    checkpoint = torch.load("pretrained.ckpt", weights_only=False, map_location="cpu")
    model_finetune = model_from_checkpoint(checkpoint, context="finetune")
    model_finetune.restart(new_dataset_info, finetune_method="heads")

    loss_conf = OmegaConf.create({"mtt::U0": init_with_defaults(LossSpecification)})
    OmegaConf.resolve(loss_conf)

    hypers = copy.deepcopy(DEFAULT_HYPERS)
    hypers["training"]["num_epochs"] = 1
    hypers["training"]["loss"] = loss_conf
    hypers["training"]["finetune"] = _finetuning_strategy(
        "heads", inherit_heads={"mtt::U0": "energy"}
    )
    hypers["training"]["finetune"]["read_from"] = "pretrained.ckpt"

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

    # Snapshot of "mtt::U0"'s head weights right after the finetuning run: they
    # started as a copy of "energy", then diverged from it during training.
    snapshot = {
        name: param.detach().clone()
        for name, param in _target_params(model_finetune, "mtt::U0").items()
    }
    assert len(snapshot) > 0
    energy_params_after_finetune = {
        name: param.detach().clone()
        for name, param in _target_params(model_finetune, "energy").items()
    }
    mtt_u0_to_energy = {name: name.replace("mtt::U0", "energy") for name in snapshot}
    assert any(
        not torch.equal(param, energy_params_after_finetune[mtt_u0_to_energy[name]])
        for name, param in snapshot.items()
    ), "the new target's head should have diverged from 'energy' during training"

    # Restart the (still ongoing) finetuning run from the checkpoint just saved,
    # with num_epochs=0: since no actual training happens, "mtt::U0" must come out
    # exactly as it went in -- if ``inherit_heads`` were redundantly reapplied, it
    # would instead be reset to (the current) "energy"'s weights.
    checkpoint = torch.load("finetuned.ckpt", weights_only=False, map_location="cpu")
    model_restart = model_from_checkpoint(checkpoint, context="restart")
    model_restart.restart(new_dataset_info)

    hypers = copy.deepcopy(DEFAULT_HYPERS)
    hypers["training"]["num_epochs"] = 0
    hypers["training"]["loss"] = loss_conf
    hypers["training"]["finetune"] = _finetuning_strategy(
        "heads", inherit_heads={"mtt::U0": "energy"}
    )
    hypers["training"]["finetune"]["read_from"] = "finetuned.ckpt"

    trainer = trainer_from_checkpoint(
        checkpoint, context="restart", hypers=hypers["training"]
    )
    trainer.train(
        model=model_restart,
        dtype=torch.float32,
        devices=[torch.device("cpu")],
        train_datasets=[dataset],
        val_datasets=[dataset],
        checkpoint_dir=".",
    )

    for name, param in _target_params(model_restart, "mtt::U0").items():
        torch.testing.assert_close(param, snapshot[name])
