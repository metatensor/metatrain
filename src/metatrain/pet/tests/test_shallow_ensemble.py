"""Tests for the optional ``shallow_ensemble`` hyperparameter."""

import copy

import pytest
import torch
from metatomic.torch import ModelOutput, System
from omegaconf import OmegaConf

from metatrain.pet import PET, Trainer
from metatrain.utils.architectures import check_architecture_options, get_default_hypers
from metatrain.utils.data import Dataset, DatasetInfo
from metatrain.utils.data.readers import read_systems, read_targets
from metatrain.utils.data.target_info import (
    get_energy_target_info,
    get_generic_target_info,
)
from metatrain.utils.ensemble import (
    make_ensemble_members,
    uncertainty_output_name,
    validate_shallow_ensemble_hypers,
)
from metatrain.utils.hypers import init_with_defaults
from metatrain.utils.loss import LossSpecification, create_loss
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from . import DATASET_PATH, DEFAULT_HYPERS, MODEL_HYPERS


def _energy_dataset_info():
    return DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "energy": get_energy_target_info(
                "energy", {"quantity": "energy", "unit": "eV"}
            )
        },
    )


def _generic_dataset_info():
    """A general (non-energy) per-atom scalar target, matching the intended
    use case (e.g. chemical shieldings) more closely than energy does."""
    return DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "mtt::shift": get_generic_target_info(
                "mtt::shift",
                {
                    "quantity": "",
                    "unit": "",
                    "type": "scalar",
                    "num_subtargets": 1,
                    "sample_kind": "atom",
                },
            )
        },
    )


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


# ===== unit tests for metatrain.utils.ensemble =====


def test_make_ensemble_members():
    members = make_ensemble_members(lambda: torch.nn.Linear(3, 4), 3)
    assert list(members.keys()) == ["member_0", "member_1", "member_2"]
    # independent (PyTorch-default-random) initialization
    assert not torch.allclose(members["member_0"].weight, members["member_1"].weight)
    state_dict_keys = list(members.state_dict().keys())
    assert "member_0.weight" in state_dict_keys
    assert "member_2.bias" in state_dict_keys


def test_validate_shallow_ensemble_hypers():
    assert validate_shallow_ensemble_hypers(None) is None
    filled = validate_shallow_ensemble_hypers({"members": 4})
    assert filled == {"scope": "head", "members": 4}
    with pytest.raises(ValueError, match="must be > 1"):
        validate_shallow_ensemble_hypers({"members": 1})
    with pytest.raises(ValueError, match="must be > 1"):
        validate_shallow_ensemble_hypers({})  # members defaults to 1


def test_uncertainty_output_name():
    assert uncertainty_output_name("energy") == "energy_uncertainty"
    assert uncertainty_output_name("mtt::foo") == "mtt::aux::foo_uncertainty"


def test_members_one_rejected_at_options_validation():
    """A ``shallow_ensemble`` block with ``members: 1`` is a hard validation
    error at options-parsing time, not a silent no-op."""
    options = OmegaConf.merge(
        {"architecture": get_default_hypers("pet")},
        {
            "architecture": {
                "name": "pet",
                "model": {"shallow_ensemble": {"members": 1}},
            }
        },
    )
    with pytest.raises(Exception, match="members"):
        check_architecture_options(
            name="pet", options=OmegaConf.to_container(options["architecture"])
        )


# ===== unit test for the ensemble_nll loss =====


def test_ensemble_nll_loss_matches_gaussian_nll_formula():
    from metatensor.torch import Labels, TensorBlock, TensorMap

    loss_fn = create_loss(
        "ensemble_nll", name="energy", gradient=None, weight=1.0, reduction="mean"
    )

    def mktmap(values):
        return TensorMap(
            keys=Labels.single(),
            blocks=[
                TensorBlock(
                    values=values,
                    samples=Labels(
                        ["system"], torch.arange(values.shape[0]).reshape(-1, 1)
                    ),
                    components=[],
                    properties=Labels(["energy"], torch.tensor([[0]])),
                )
            ],
        )

    mean = mktmap(torch.tensor([[1.0], [2.0], [3.0]]))
    target = mktmap(torch.tensor([[1.5], [2.0], [2.5]]))
    var = mktmap(torch.tensor([[0.25], [0.25], [0.25]]))

    loss = loss_fn(
        {"energy": mean, "energy_uncertainty": var},
        {"energy": target},
    )
    resid = (mean.block().values - target.block().values).squeeze(-1)
    v = var.block().values.squeeze(-1)
    expected = (0.5 * (resid**2 / v + torch.log(v))).mean()
    torch.testing.assert_close(loss, expected)

    with pytest.raises(ValueError, match="requires"):
        loss_fn({"energy": mean}, {"energy": target})  # missing uncertainty


# ===== integration tests: forward pass, both scopes, both target kinds =====


@pytest.mark.parametrize("scope", ["head", "readout"])
@pytest.mark.parametrize("target_kind", ["energy", "generic"])
def test_shallow_ensemble_mean_and_uncertainty(scope, target_kind):
    if target_kind == "energy":
        dataset_info = _energy_dataset_info()
        target_name, unc_name, sample_kind = "energy", "energy_uncertainty", "system"
    else:
        dataset_info = _generic_dataset_info()
        target_name = "mtt::shift"
        unc_name = "mtt::aux::shift_uncertainty"
        sample_kind = "atom"

    hypers = copy.deepcopy(MODEL_HYPERS)
    hypers["shallow_ensemble"] = {"scope": scope, "members": 3}
    model = PET(hypers, dataset_info)
    model.eval()

    # member_{e} shows up as a state-dict path segment
    member_keys = [k for k in model.state_dict().keys() if ".member_" in k]
    assert len(member_keys) > 0

    system = _make_system(model)
    outputs = {
        target_name: ModelOutput(sample_kind=sample_kind),
        unc_name: ModelOutput(sample_kind=sample_kind),
    }
    out = model([system], outputs)

    mean = out[target_name].block().values
    uncertainty = out[unc_name].block().values
    assert uncertainty.shape == mean.shape
    assert torch.isfinite(mean).all()
    assert torch.isfinite(uncertainty).all()
    assert (uncertainty >= 0).all()


@pytest.mark.parametrize("scope", ["head", "readout"])
def test_shallow_ensemble_padding_invariant(scope):
    hypers = copy.deepcopy(MODEL_HYPERS)
    hypers["shallow_ensemble"] = {"scope": scope, "members": 3}
    model = PET(hypers, _energy_dataset_info())
    model.eval()

    outputs = {
        "energy": ModelOutput(sample_kind="system"),
        "energy_uncertainty": ModelOutput(sample_kind="system"),
    }

    system = _make_system(model)
    lone = model([system], outputs)

    system2 = System(
        types=torch.tensor([6] * 7),
        positions=torch.stack([torch.tensor([0.0, 0.0, float(i)]) for i in range(7)]),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )
    system2 = get_system_with_neighbor_lists(system2, model.requested_neighbor_lists())
    padded = model([system, system2], outputs)

    torch.testing.assert_close(
        lone["energy"].block().values[0], padded["energy"].block().values[0]
    )
    torch.testing.assert_close(
        lone["energy_uncertainty"].block().values[0],
        padded["energy_uncertainty"].block().values[0],
    )


def test_shallow_ensemble_torchscript():
    hypers = copy.deepcopy(MODEL_HYPERS)
    hypers["shallow_ensemble"] = {"scope": "head", "members": 2}
    model = PET(hypers, _energy_dataset_info())
    model.eval()
    system = _make_system(model)
    outputs = {
        "energy": ModelOutput(sample_kind="system"),
        "energy_uncertainty": ModelOutput(sample_kind="system"),
    }
    eager_out = model([system], outputs)
    scripted = torch.jit.script(model)
    scripted_out = scripted([system], outputs)
    torch.testing.assert_close(
        eager_out["energy"].block().values, scripted_out["energy"].block().values
    )
    torch.testing.assert_close(
        eager_out["energy_uncertainty"].block().values,
        scripted_out["energy_uncertainty"].block().values,
    )


# ===== integration tests: training with both loss modes =====


def _train_tiny_model(loss_type, scope="head", members=3):
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

    hypers = copy.deepcopy(DEFAULT_HYPERS)
    hypers["model"]["shallow_ensemble"] = {"scope": scope, "members": members}
    hypers["model"]["d_pet"] = 8
    hypers["model"]["d_node"] = 8
    hypers["model"]["d_feedforward"] = 8
    hypers["model"]["num_heads"] = 2
    hypers["model"]["num_gnn_layers"] = 1
    hypers["model"]["num_attention_layers"] = 1
    hypers["model"]["num_head_layers"] = 1
    hypers["training"]["num_epochs"] = 2
    hypers["training"]["num_workers"] = 0
    hypers["training"]["batch_size"] = 4
    loss_conf = init_with_defaults(LossSpecification)
    loss_conf["type"] = loss_type
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


@pytest.mark.parametrize("loss_type", ["mse", "ensemble_nll"])
def test_shallow_ensemble_trains(loss_type):
    """A tiny end-to-end training run completes without error for both
    supported loss modes: a plain loss on the mean (diversity from init only),
    and the ensemble-aware NLL (diversity encouraged by scoring the spread)."""
    model = _train_tiny_model(loss_type)

    model.eval()
    system = _make_system(model)
    outputs = {
        "energy": ModelOutput(sample_kind="system"),
        "energy_uncertainty": ModelOutput(sample_kind="system"),
    }
    out = model([system], outputs)
    assert torch.isfinite(out["energy"].block().values).all()
    assert torch.isfinite(out["energy_uncertainty"].block().values).all()
    assert (out["energy_uncertainty"].block().values >= 0).all()
