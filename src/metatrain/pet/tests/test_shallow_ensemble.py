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
    assert filled == {"scope": "head", "members": 4, "dropout": 0.0, "bagging": 1.0}
    with pytest.raises(ValueError, match="must be > 1"):
        validate_shallow_ensemble_hypers({"members": 1})
    with pytest.raises(ValueError, match="must be > 1"):
        validate_shallow_ensemble_hypers({})  # members defaults to 1


@pytest.mark.parametrize("dropout", [-0.1, 1.0, 1.5])
def test_dropout_out_of_range_rejected(dropout):
    with pytest.raises(ValueError, match=r"dropout' must be in \[0, 1\)"):
        validate_shallow_ensemble_hypers({"members": 4, "dropout": dropout})


@pytest.mark.parametrize("bagging", [0.0, -0.1, 1.5])
def test_bagging_out_of_range_rejected(bagging):
    with pytest.raises(ValueError, match=r"bagging' must be in \(0, 1\]"):
        validate_shallow_ensemble_hypers({"members": 4, "bagging": bagging})


def test_dropout_with_readout_scope_rejected():
    with pytest.raises(ValueError, match="no effect with scope='readout'"):
        validate_shallow_ensemble_hypers(
            {"scope": "readout", "members": 4, "dropout": 0.1}
        )
    # dropout=0 (the default) with scope="readout" is fine -- it's a no-op either way
    filled = validate_shallow_ensemble_hypers({"scope": "readout", "members": 4})
    assert filled["dropout"] == 0.0


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


# ===== integration tests: dropout and bagging =====


def _make_ensemble_model(scope="head", members=3, dropout=0.0, bagging=1.0):
    hypers = copy.deepcopy(MODEL_HYPERS)
    hypers["shallow_ensemble"] = {
        "scope": scope,
        "members": members,
        "dropout": dropout,
        "bagging": bagging,
    }
    return PET(hypers, _energy_dataset_info())


_ENERGY_OUTPUTS = {
    "energy": ModelOutput(sample_kind="system"),
    "energy_uncertainty": ModelOutput(sample_kind="system"),
}


def test_dropout_leaves_structure_unchanged_when_zero():
    """dropout=0 (the default) must not insert any Dropout module at all, so an
    existing config (or checkpoint) that never mentions dropout sees exactly the
    same model structure as before this hyperparameter was added."""
    with_field = _make_ensemble_model(dropout=0.0)
    without_field = PET(
        {
            **copy.deepcopy(MODEL_HYPERS),
            "shallow_ensemble": {"scope": "head", "members": 3},
        },
        _energy_dataset_info(),
    )
    assert list(with_field.state_dict().keys()) == list(
        without_field.state_dict().keys()
    )
    assert not any(isinstance(m, torch.nn.Dropout) for m in with_field.modules())


def test_dropout_present_when_nonzero():
    model = _make_ensemble_model(dropout=0.2)
    assert any(isinstance(m, torch.nn.Dropout) for m in model.modules())


def test_dropout_is_stochastic_in_train_and_deterministic_in_eval():
    model = _make_ensemble_model(dropout=0.3, members=4)
    system = _make_system(model)

    model.train()
    torch.manual_seed(0)
    out_a = model([system], _ENERGY_OUTPUTS)["energy"].block().values.clone()
    torch.manual_seed(1)
    out_b = model([system], _ENERGY_OUTPUTS)["energy"].block().values.clone()
    assert not torch.allclose(out_a, out_b), (
        "different dropout masks should give different mean predictions in train mode"
    )

    model.eval()
    out_c = model([system], _ENERGY_OUTPUTS)["energy"].block().values.clone()
    out_d = model([system], _ENERGY_OUTPUTS)["energy"].block().values.clone()
    torch.testing.assert_close(out_c, out_d)


def test_bagging_off_matches_plain_mean_even_in_train_mode():
    """bagging=1.0 (the default) must be a pure no-op: the same computation as
    before this hyperparameter existed, in train mode too."""
    model = _make_ensemble_model(bagging=1.0, members=4)
    model.train()
    system = _make_system(model)
    torch.manual_seed(0)
    out_a = model([system], _ENERGY_OUTPUTS)["energy"].block().values.clone()
    torch.manual_seed(1234)
    out_b = model([system], _ENERGY_OUTPUTS)["energy"].block().values.clone()
    # no dropout here, and bagging is off, so train-mode is fully deterministic
    torch.testing.assert_close(out_a, out_b)


def test_bagging_is_stochastic_in_train_and_deterministic_in_eval():
    model = _make_ensemble_model(bagging=0.5, members=4)
    system = _make_system(model)

    model.train()
    torch.manual_seed(0)
    out_a = model([system], _ENERGY_OUTPUTS)["energy"].block().values.clone()
    torch.manual_seed(1)
    out_b = model([system], _ENERGY_OUTPUTS)["energy"].block().values.clone()
    assert not torch.allclose(out_a, out_b), (
        "different bagging draws should give different mean predictions in train mode"
    )

    model.eval()
    out_c = model([system], _ENERGY_OUTPUTS)["energy"].block().values.clone()
    out_d = model([system], _ENERGY_OUTPUTS)["energy"].block().values.clone()
    torch.testing.assert_close(out_c, out_d)


def test_bagging_does_not_affect_reported_uncertainty():
    """The variance must stay the plain, unweighted spread across members,
    regardless of the bagging draw -- only the mean that reaches the loss is
    reweighted."""
    model = _make_ensemble_model(bagging=0.5, members=4)
    model.train()
    system = _make_system(model)

    torch.manual_seed(0)
    unc_a = (
        model([system], _ENERGY_OUTPUTS)["energy_uncertainty"].block().values.clone()
    )
    torch.manual_seed(1)
    unc_b = (
        model([system], _ENERGY_OUTPUTS)["energy_uncertainty"].block().values.clone()
    )
    torch.testing.assert_close(unc_a, unc_b)


def test_bagging_gradients_reach_all_members():
    """Even with a harsh keep probability, backprop through the bagged mean
    must still reach every member's parameters over enough forward passes (no
    member silently starved of gradient by construction)."""
    model = _make_ensemble_model(bagging=0.3, members=4, scope="head")
    model.train()
    system = _make_system(model)

    member_params = [
        p
        for name, p in model.named_parameters()
        if ".member_" in name and p.requires_grad
    ]
    assert len(member_params) > 0
    seen_grad = [False] * len(member_params)
    for seed in range(30):
        model.zero_grad()
        torch.manual_seed(seed)
        out = model([system], {"energy": ModelOutput(sample_kind="system")})
        out["energy"].block().values.sum().backward()
        for i, p in enumerate(member_params):
            if p.grad is not None and torch.any(p.grad != 0.0):
                seen_grad[i] = True
    assert all(seen_grad), "some member never received a nonzero gradient"


def test_shallow_ensemble_torchscript_with_dropout_and_bagging():
    model = _make_ensemble_model(dropout=0.2, bagging=0.5, members=2)
    model.eval()
    system = _make_system(model)
    eager_out = model([system], _ENERGY_OUTPUTS)
    scripted = torch.jit.script(model)
    scripted_out = scripted([system], _ENERGY_OUTPUTS)
    # eval mode: dropout/bagging are inactive either way, so this is a plain
    # eager-vs-scripted equivalence check, exactly like the plain test above
    torch.testing.assert_close(
        eager_out["energy"].block().values, scripted_out["energy"].block().values
    )
    torch.testing.assert_close(
        eager_out["energy_uncertainty"].block().values,
        scripted_out["energy_uncertainty"].block().values,
    )


# ===== integration tests: training with both loss modes =====


def _train_tiny_model(loss_type, scope="head", members=3, dropout=0.0, bagging=1.0):
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
    hypers["model"]["shallow_ensemble"] = {
        "scope": scope,
        "members": members,
        "dropout": dropout,
        "bagging": bagging,
    }
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


@pytest.mark.parametrize("loss_type", ["mse", "ensemble_nll"])
def test_shallow_ensemble_trains_with_dropout_and_bagging(loss_type):
    """As above, with both new diversity mechanisms active together."""
    model = _train_tiny_model(loss_type, dropout=0.1, bagging=0.7)

    model.eval()
    system = _make_system(model)
    outputs = {
        "energy": ModelOutput(sample_kind="system"),
        "energy_uncertainty": ModelOutput(sample_kind="system"),
    }
    # eval mode: dropout/bagging are inactive, so this is deterministic
    out_a = model([system], outputs)
    out_b = model([system], outputs)
    torch.testing.assert_close(
        out_a["energy"].block().values, out_b["energy"].block().values
    )
    assert torch.isfinite(out_a["energy"].block().values).all()
    assert torch.isfinite(out_a["energy_uncertainty"].block().values).all()
    assert (out_a["energy_uncertainty"].block().values >= 0).all()
