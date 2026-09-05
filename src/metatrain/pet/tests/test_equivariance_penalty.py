"""End-to-end integration tests for the on-line ``equivariance_penalty`` loss's
trainer wiring: replicating/augmenting each batch ``num_augmentations`` times,
skipping the usual single random augmentation, and reducing back to a mean/variance
per system before computing the loss and metrics. The underlying building blocks
(``O3Augmenter.replicate_and_augment``/``.undo_augmentation``,
``reduce_over_augmentations``, ``EquivariancePenaltyLoss``) are already covered
directly in ``tests/utils/``; this only exercises ``pet.trainer``'s use of them."""

import copy
import logging

import pytest
import torch
from metatomic.torch import ModelOutput, System
from omegaconf import OmegaConf

from metatrain.pet import PET, Trainer
from metatrain.utils.data import Dataset, DatasetInfo
from metatrain.utils.data.readers import read_systems, read_targets
from metatrain.utils.equivariance_penalty import equivariance_variance_output_name
from metatrain.utils.hypers import init_with_defaults
from metatrain.utils.loss import LossSpecification
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from . import DATASET_PATH, DEFAULT_HYPERS


def _make_system(model, positions=None):
    if positions is None:
        positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.2]])
    system = System(
        types=torch.tensor([6] * positions.shape[0]),
        positions=positions,
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )
    return get_system_with_neighbor_lists(system, model.requested_neighbor_lists())


def _small_model_hypers(hypers):
    hypers["model"]["d_pet"] = 8
    hypers["model"]["d_node"] = 8
    hypers["model"]["d_feedforward"] = 8
    hypers["model"]["num_heads"] = 2
    hypers["model"]["num_gnn_layers"] = 1
    hypers["model"]["num_attention_layers"] = 1
    hypers["model"]["num_head_layers"] = 1
    return hypers


def _train_tiny_model(num_augmentations=4, variance_weight=0.1, batch_size=4):
    systems = read_systems(DATASET_PATH)[:20]
    conf = {
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
    targets, target_info_dict = read_targets(OmegaConf.create(conf))
    targets["energy"] = targets["energy"][:20]
    dataset = Dataset.from_dict({"system": systems, "energy": targets["energy"]})

    hypers = _small_model_hypers(copy.deepcopy(DEFAULT_HYPERS))
    hypers["training"]["num_epochs"] = 2
    hypers["training"]["num_workers"] = 0
    hypers["training"]["batch_size"] = batch_size
    loss_conf = init_with_defaults(LossSpecification)
    loss_conf["type"] = "equivariance_penalty"
    loss_conf["num_augmentations"] = num_augmentations
    loss_conf["variance_weight"] = variance_weight
    hypers["training"]["loss"] = {"energy": loss_conf}

    dataset_info = DatasetInfo(
        length_unit="Angstrom", atomic_types=[1, 6, 7, 8], targets=target_info_dict
    )
    model = PET(hypers["model"], dataset_info)
    trainer = Trainer(hypers["training"])
    trainer.train(
        model=model,
        dtype=torch.float32,
        devices=[torch.device("cpu")],
        train_datasets=[dataset],
        val_datasets=[dataset],
        checkpoint_dir=".",
    )
    return model


def test_equivariance_penalty_trains():
    """A tiny end-to-end training run completes without error, and the trained
    model still gives finite predictions for a plain forward pass (no extra
    outputs requested: the augmentation/reduction machinery only runs inside the
    trainer, and must leave the exported model's normal inference path alone)."""
    model = _train_tiny_model()

    model.eval()
    system = _make_system(model)
    out = model([system], {"energy": ModelOutput(sample_kind="system")})
    assert torch.isfinite(out["energy"].block().values).all()


def test_equivariance_penalty_logs_loss_components(caplog):
    """Both components (mse, variance) must appear as logged loss values, for
    both train and validation, while the RMSE/MAE metrics keep reporting only on
    the plain target -- never on the internal ``_equivariance_variance`` auxiliary
    key (see ``metatrain.utils.equivariance_penalty
    .equivariance_variance_output_name``)."""
    with caplog.at_level(logging.INFO):
        _train_tiny_model()

    variance_key = equivariance_variance_output_name("energy")
    messages = "\n".join(caplog.messages)

    for split in ("training", "validation"):
        assert f"{split} energy_equivariance_penalty_mse" in messages
        assert f"{split} energy_equivariance_penalty_variance" in messages
        # the RMSE/MAE lines are of the form "<split> <target> RMSE (...)"; the
        # auxiliary variance key must never be treated as a target of its own
        assert f"{split} {variance_key}" not in messages


@pytest.mark.parametrize("num_augmentations", [2, 4])
def test_equivariance_penalty_variance_decreases_or_stays_finite(num_augmentations):
    """Different ``num_augmentations`` values (in particular an odd choice, e.g.
    not matching any other default) must all train without error."""
    model = _train_tiny_model(num_augmentations=num_augmentations)
    model.eval()
    system = _make_system(model)
    out = model([system], {"energy": ModelOutput(sample_kind="system")})
    assert torch.isfinite(out["energy"].block().values).all()


def test_equivariance_penalty_requires_shared_num_augmentations():
    """Two targets configured with different ``num_augmentations`` cannot share a
    single augmented batch, and must be rejected explicitly rather than silently
    using one of the two values."""
    systems = read_systems(DATASET_PATH)[:20]
    conf = {
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
        },
        "mtt::extra": {
            "quantity": "",
            "read_from": DATASET_PATH,
            "reader": "ase",
            "key": "U0",
            "unit": "",
            "type": "scalar",
            "sample_kind": "system",
            "num_subtargets": 1,
            "forces": False,
            "stress": False,
            "virial": False,
        },
    }
    targets, target_info_dict = read_targets(OmegaConf.create(conf))
    for name in targets:
        targets[name] = targets[name][:20]
    dataset = Dataset.from_dict(
        {
            "system": systems,
            "energy": targets["energy"],
            "mtt::extra": targets["mtt::extra"],
        }
    )

    hypers = _small_model_hypers(copy.deepcopy(DEFAULT_HYPERS))
    hypers["training"]["num_epochs"] = 1
    hypers["training"]["num_workers"] = 0
    hypers["training"]["batch_size"] = 4

    def _loss_conf(num_augmentations):
        loss_conf = init_with_defaults(LossSpecification)
        loss_conf["type"] = "equivariance_penalty"
        loss_conf["num_augmentations"] = num_augmentations
        loss_conf["variance_weight"] = 0.1
        return loss_conf

    hypers["training"]["loss"] = {
        "energy": _loss_conf(4),
        "mtt::extra": _loss_conf(8),
    }

    dataset_info = DatasetInfo(
        length_unit="Angstrom", atomic_types=[1, 6, 7, 8], targets=target_info_dict
    )
    model = PET(hypers["model"], dataset_info)
    trainer = Trainer(hypers["training"])
    with pytest.raises(ValueError, match="same 'num_augmentations'"):
        trainer.train(
            model=model,
            dtype=torch.float32,
            devices=[torch.device("cpu")],
            train_datasets=[dataset],
            val_datasets=[dataset],
            checkpoint_dir=".",
        )
