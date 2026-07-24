import copy
import shutil

import pytest
import torch
from omegaconf import OmegaConf

from metatrain.experimental.space import SPACE, Trainer
from metatrain.experimental.space.modules.finetuning import apply_finetuning_strategy
from metatrain.utils.data import Dataset, DatasetInfo
from metatrain.utils.data.readers import read_systems, read_targets
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.hypers import init_with_defaults
from metatrain.utils.io import model_from_checkpoint, trainer_from_checkpoint
from metatrain.utils.loss import LossSpecification

from . import DATASET_PATH, DEFAULT_HYPERS, MODEL_HYPERS


def _model():
    target_info_dict = {
        "energy": get_energy_target_info("energy", {"quantity": "energy", "unit": "eV"})
    }
    dataset_info = DatasetInfo(
        length_unit="Angstrom", atomic_types=[1, 6, 7, 8], targets=target_info_dict
    )
    return SPACE(MODEL_HYPERS, dataset_info)


def test_lora_finetuning_default_modules():
    """LoRA with an empty config must resolve to SPACE's own module names and
    actually inject adapters; the shared PET defaults match nothing here."""
    model = _model()

    finetuning_strategy = {
        "read_from": None,
        "method": "lora",
        "config": {"rank": 4, "alpha": 8},
        "inherit_heads": {},
    }

    model = apply_finetuning_strategy(model, finetuning_strategy)

    assert any("lora_" in name for name, _ in model.named_parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = sum(p.numel() for p in model.parameters())
    assert 0 < num_trainable_params < num_params

    # the resolved, not the user-written, config is what gets stored, so that
    # reloading a finetuned checkpoint targets the same modules
    assert model.finetune_config["config"]["target_modules"] == ["linear_layer"]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_lora_finetuning_device(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    model = _model().to(device)

    finetuning_strategy = {
        "read_from": None,
        "method": "lora",
        "config": {"rank": 4, "alpha": 8},
        "inherit_heads": {},
    }

    model = apply_finetuning_strategy(model, finetuning_strategy)
    for param in model.parameters():
        assert param.device.type == device


def test_heads_finetuning_default_modules():
    """``heads`` with an empty config must freeze the backbone but leave the
    SPACE heads and last layers trainable."""
    model = _model()

    finetuning_strategy = {
        "read_from": None,
        "method": "heads",
        "config": {},
        "inherit_heads": {},
    }

    model = apply_finetuning_strategy(model, finetuning_strategy)

    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable, "no parameters left trainable"
    assert all(".heads." in name or ".last_layers." in name for name in trainable), (
        "backbone parameters were left trainable"
    )

    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = sum(p.numel() for p in model.parameters())
    assert 0 < num_trainable_params < num_params


def test_heads_finetuning_unknown_modules():
    """Unknown 'head_modules'/'last_layer_modules' should raise, not silently
    freeze the whole model."""
    model = _model()

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
    """A LoRA-finetuned checkpoint must round-trip: the adapters have to survive
    saving and reloading, otherwise the restarted run silently trains the wrong
    parameter set."""

    monkeypatch.chdir(tmp_path)
    shutil.copy(DATASET_PATH, "qm9_reduced_100.xyz")

    systems = read_systems(DATASET_PATH)

    target_info_dict = {
        "mtt::U0": get_energy_target_info(
            "mtt::U0", {"quantity": "energy", "unit": "eV"}
        )
    }

    dataset_info = DatasetInfo(
        length_unit="Angstrom", atomic_types=[1, 6, 7, 8], targets=target_info_dict
    )
    model = SPACE(MODEL_HYPERS, dataset_info)

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

    loss_conf = OmegaConf.create({"mtt::U0": init_with_defaults(LossSpecification)})
    OmegaConf.resolve(loss_conf)

    hypers = copy.deepcopy(DEFAULT_HYPERS)
    hypers["training"]["num_epochs"] = 1
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
    assert isinstance(model_finetune, SPACE)
    model_finetune.restart(dataset_info)

    hypers = copy.deepcopy(DEFAULT_HYPERS)
    hypers["training"]["num_epochs"] = 0
    hypers["training"]["loss"] = loss_conf
    hypers["training"]["finetune"] = {
        "read_from": "tmp.ckpt",
        "method": "lora",
        "config": {"rank": 4, "alpha": 8},
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

    assert any("lora_" in name for name, _ in model_finetune.named_parameters())

    # Finetuning restart
    checkpoint = torch.load("finetuned.ckpt", weights_only=False, map_location="cpu")
    model_finetune_restart = model_from_checkpoint(checkpoint, context="restart")
    assert isinstance(model_finetune_restart, SPACE)
    model_finetune_restart.restart(dataset_info)

    assert any("lora_" in name for name, _ in model_finetune_restart.named_parameters())


def _target_params(model, target_name):
    return {
        name: param
        for name, param in model.named_parameters()
        if f".{target_name}." in name
    }


def _energy_conf(target_name):
    return {
        target_name: {
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


def _finetune_strategy(method, read_from=None, inherit_heads=None):
    # an empty config is what exercises SPACE's own default module names
    return {
        "read_from": read_from,
        "method": method,
        "config": {},
        "inherit_heads": inherit_heads or {},
    }


def _two_target_setup():
    """A model carrying both an ``"energy"`` and an ``"mtt::U0"`` head, plus the
    dataset info of a run that only trains on ``"mtt::U0"``, leaving ``"energy"``
    stale.

    SPACE's ``restart`` cannot grow a head for a target the model was not built
    with, so both heads have to exist from the start -- unlike PET, where the
    equivalent setup adds the second target on restart."""
    targets = {
        name: get_energy_target_info(name, {"quantity": "energy", "unit": "eV"})
        for name in ("energy", "mtt::U0")
    }
    model = SPACE(
        MODEL_HYPERS,
        DatasetInfo(length_unit="Angstrom", atomic_types=[1, 6, 7, 8], targets=targets),
    )
    new_dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={"mtt::U0": targets["mtt::U0"]},
    )
    return model, new_dataset_info


def _assert_target_absent(model, target_name):
    assert target_name not in model.dataset_info.targets
    assert target_name not in model.target_names
    assert target_name not in model.supported_outputs()
    assert target_name not in model.module.module.heads
    assert target_name not in model.module.module.last_layers
    assert target_name not in model.key_labels
    assert target_name not in model.component_labels
    assert target_name not in model.property_labels
    for additive_model in model.additive_models:
        assert target_name not in additive_model.outputs
    assert target_name not in model.scaler.outputs


def _assert_target_present(model, target_name):
    assert target_name in model.dataset_info.targets
    assert target_name in model.target_names
    assert target_name in model.supported_outputs()
    assert target_name in model.module.module.heads
    assert target_name in model.module.module.last_layers
    assert target_name in model.key_labels


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

    model.restart(new_dataset_info)
    _assert_target_present(model, "energy")

    apply_finetuning_strategy(model, _finetune_strategy(method))

    _assert_target_absent(model, "energy")
    _assert_target_present(model, "mtt::U0")


def test_finetune_full_inherit_heads_then_prunes_source_target():
    """``inherit_heads`` can copy weights from a stale target's head into the new
    target's head; the stale target is only removed afterwards."""
    model, new_dataset_info = _two_target_setup()

    model.restart(new_dataset_info)
    apply_finetuning_strategy(
        model, _finetune_strategy("full", inherit_heads={"mtt::U0": "energy"})
    )

    _assert_target_absent(model, "energy")
    _assert_target_present(model, "mtt::U0")


def test_finetune_heads_keeps_stale_targets():
    """With heads-only finetuning, the backbone is unchanged, so a target not part
    of the current run's dataset must be kept."""
    model, new_dataset_info = _two_target_setup()

    model.restart(new_dataset_info)
    apply_finetuning_strategy(model, _finetune_strategy("heads"))

    _assert_target_present(model, "energy")
    _assert_target_present(model, "mtt::U0")


def test_plain_restart_keeps_stale_targets():
    """A plain restart (not part of a finetuning run) must not prune any target."""
    model, new_dataset_info = _two_target_setup()

    model.restart(new_dataset_info)

    _assert_target_present(model, "energy")
    _assert_target_present(model, "mtt::U0")


def test_finetuning_restart_does_not_reapply_inherit_heads(monkeypatch, tmp_path):
    """Restarting an interrupted finetuning run must not redo the one-time
    ``inherit_heads`` weight copy: the destination head's weights, once trained,
    should survive a restart with ``num_epochs=0`` unchanged. Redoing the copy
    would silently reset them to the source head and throw away the progress."""

    monkeypatch.chdir(tmp_path)
    shutil.copy(DATASET_PATH, "qm9_reduced_100.xyz")

    systems = [system.to(torch.float64) for system in read_systems(DATASET_PATH)]

    def _dataset(target_name):
        targets, _ = read_targets(OmegaConf.create(_energy_conf(target_name)))
        return Dataset.from_dict({"system": systems, target_name: targets[target_name]})

    def _loss(*target_names):
        loss_conf = OmegaConf.create(
            {name: init_with_defaults(LossSpecification) for name in target_names}
        )
        OmegaConf.resolve(loss_conf)
        return loss_conf

    # Both heads have to exist up front: SPACE's ``restart`` cannot grow a head
    # for a target the model was not built with, so ``inherit_heads`` can only
    # ever copy between targets that are already there.
    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            name: get_energy_target_info(name, {"quantity": "energy", "unit": "eV"})
            for name in ("energy", "mtt::U0")
        },
    )
    both_datasets = [_dataset("energy"), _dataset("mtt::U0")]

    # Pre-training on both targets.
    model = SPACE(MODEL_HYPERS, dataset_info)
    hypers = copy.deepcopy(DEFAULT_HYPERS)
    hypers["training"]["num_epochs"] = 1
    hypers["training"]["loss"] = _loss("energy", "mtt::U0")

    trainer = Trainer(hypers["training"])
    trainer.train(
        model=model,
        dtype=torch.float32,
        devices=[torch.device("cpu")],
        train_datasets=both_datasets,
        val_datasets=both_datasets,
        checkpoint_dir=".",
    )
    trainer.save_checkpoint(model, "pretrained.ckpt")

    # Finetune "mtt::U0", re-initializing its head from "energy". ``heads`` keeps
    # the backbone frozen, and only "mtt::U0" is in this run's loss, so its head
    # is the only thing that moves.
    checkpoint = torch.load("pretrained.ckpt", weights_only=False, map_location="cpu")
    model_finetune = model_from_checkpoint(checkpoint, context="finetune")
    model_finetune.restart(dataset_info)

    hypers = copy.deepcopy(DEFAULT_HYPERS)
    hypers["training"]["num_epochs"] = 1
    hypers["training"]["loss"] = _loss("mtt::U0")
    hypers["training"]["finetune"] = _finetune_strategy(
        "heads", read_from="pretrained.ckpt", inherit_heads={"mtt::U0": "energy"}
    )

    trainer = Trainer(hypers["training"])
    trainer.train(
        model=model_finetune,
        dtype=torch.float32,
        devices=[torch.device("cpu")],
        train_datasets=[_dataset("mtt::U0")],
        val_datasets=[_dataset("mtt::U0")],
        checkpoint_dir=".",
    )
    trainer.save_checkpoint(model_finetune, "finetuned.ckpt")

    # "mtt::U0"'s head started this run as a copy of "energy" and then diverged
    # from it by training; without that divergence the assertion below could not
    # tell a preserved head from a re-copied one.
    snapshot = {
        name: param.detach().clone()
        for name, param in _target_params(model_finetune, "mtt::U0").items()
    }
    assert len(snapshot) > 0
    energy_params = _target_params(model_finetune, "energy")
    assert any(
        not torch.equal(param, energy_params[name.replace("mtt::U0", "energy")])
        for name, param in snapshot.items()
    ), "the finetuned head should have diverged from 'energy' during training"

    # Restart that still-ongoing finetuning run with num_epochs=0: nothing trains,
    # so "mtt::U0" must come out exactly as it went in.
    checkpoint = torch.load("finetuned.ckpt", weights_only=False, map_location="cpu")
    model_restart = model_from_checkpoint(checkpoint, context="restart")
    model_restart.restart(dataset_info)

    hypers = copy.deepcopy(DEFAULT_HYPERS)
    hypers["training"]["num_epochs"] = 0
    hypers["training"]["loss"] = _loss("mtt::U0")
    hypers["training"]["finetune"] = _finetune_strategy(
        "heads", read_from="finetuned.ckpt", inherit_heads={"mtt::U0": "energy"}
    )

    trainer = trainer_from_checkpoint(
        checkpoint, context="restart", hypers=hypers["training"]
    )
    trainer.train(
        model=model_restart,
        dtype=torch.float32,
        devices=[torch.device("cpu")],
        train_datasets=[_dataset("mtt::U0")],
        val_datasets=[_dataset("mtt::U0")],
        checkpoint_dir=".",
    )

    for name, param in _target_params(model_restart, "mtt::U0").items():
        torch.testing.assert_close(param, snapshot[name])
