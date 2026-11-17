"""Tests for ``num_neighbors_adaptive`` given as a ``[low, high]`` range."""

import copy

import pytest
import torch
from metatomic.torch import ModelOutput, System
from omegaconf import OmegaConf

import metatrain.pet.modules.backend as backend_module
from metatrain.pet import PET, Trainer
from metatrain.pet.modules.adaptive_cutoff import (
    parse_num_neighbors_adaptive,
    sample_num_neighbors_adaptive,
)
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.hypers import init_with_defaults
from metatrain.utils.loss import LossSpecification
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from . import DEFAULT_HYPERS, MODEL_HYPERS
from .test_basic import PETTests


DATASET_INFO = DatasetInfo(
    length_unit="Angstrom",
    atomic_types=[6],
    targets={
        "energy": get_energy_target_info("energy", {"quantity": "energy", "unit": "eV"})
    },
)


def _hypers(num_neighbors_adaptive):
    hypers = copy.deepcopy(MODEL_HYPERS)
    hypers["cutoff"] = 6.0
    hypers["num_neighbors_adaptive"] = num_neighbors_adaptive
    return hypers


def _system(model):
    torch.manual_seed(1)
    system = System(
        types=torch.full((12,), 6),
        positions=torch.rand(12, 3) * 5.0,
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )
    return get_system_with_neighbor_lists(system, model.requested_neighbor_lists())


@pytest.fixture
def recorded_targets(monkeypatch):
    """Record the target number of neighbors passed to ``compute_batch_tensors``."""
    recorded = []
    original = backend_module.compute_batch_tensors

    def wrapper(*args, **kwargs):
        recorded.append(args[10])
        return original(*args, **kwargs)

    monkeypatch.setattr(backend_module, "compute_batch_tensors", wrapper)
    return recorded


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, None),
        (8, (8, 8)),
        ([4, 12], (4, 12)),
        ((4, 12), (4, 12)),
        ([7, 7], (7, 7)),
    ],
)
def test_parse(value, expected):
    assert parse_num_neighbors_adaptive(value) == expected


@pytest.mark.parametrize("value", [[4], [4, 8, 12], [12, 4], 0, [0, 4]])
def test_parse_invalid(value):
    with pytest.raises(ValueError):
        parse_num_neighbors_adaptive(value)


def test_sample():
    torch.manual_seed(0)
    samples = {sample_num_neighbors_adaptive(4, 7) for _ in range(200)}
    assert samples == {4.0, 5.0, 6.0, 7.0}

    # a degenerate range does not consume random numbers
    state = torch.get_rng_state()
    assert sample_num_neighbors_adaptive(5, 5) == 5.0
    assert torch.equal(state, torch.get_rng_state())


def test_training_samples_from_range(recorded_targets):
    torch.manual_seed(0)
    model = PET(_hypers([4, 12]), DATASET_INFO)
    system = _system(model)
    outputs = {"energy": ModelOutput(sample_kind="system")}

    model.train()
    for _ in range(50):
        model([system], outputs)
    assert all(v == int(v) and 4 <= v <= 12 for v in recorded_targets)
    assert len(set(recorded_targets)) > 1

    recorded_targets.clear()
    model.eval()
    model([system], outputs)
    assert recorded_targets == [4.0]


def test_eval_matches_fixed_model():
    """In eval mode, a range model behaves as a fixed model with ``low``."""
    torch.manual_seed(0)
    range_model = PET(_hypers([4, 12]), DATASET_INFO).eval()
    fixed_model = PET(_hypers(4), DATASET_INFO).eval()
    fixed_model.load_state_dict(range_model.state_dict())

    system = _system(range_model)
    outputs = {
        "energy": ModelOutput(sample_kind="system"),
        "mtt::aux::cutoff_stats": ModelOutput(sample_kind="atom"),
    }
    range_out = range_model([system], outputs)
    fixed_out = fixed_model([system], outputs)
    for key in outputs:
        torch.testing.assert_close(
            range_out[key].block().values, fixed_out[key].block().values
        )


def test_change_num_neighbors_adaptive(recorded_targets):
    model = PET(_hypers([4, 12]), DATASET_INFO).eval()
    system = _system(model)
    outputs = {"energy": ModelOutput(sample_kind="system")}

    model.set_num_neighbors_adaptive(10)
    assert model.hypers["num_neighbors_adaptive"] == 10
    model([system], outputs)
    assert recorded_targets == [10.0]

    # restarting with a different value is allowed
    recorded_targets.clear()
    model.restart(DATASET_INFO, _hypers([6, 8]))
    assert model.hypers["num_neighbors_adaptive"] == [6, 8]
    model([system], outputs)
    model.train()
    for _ in range(20):
        model([system], outputs)
    assert recorded_targets[0] == 6.0
    assert set(recorded_targets[1:]) <= {6.0, 7.0, 8.0}
    model.eval()

    # the value is kept in the checkpoint
    loaded = PET.load_checkpoint(model.get_checkpoint(), context="export")
    assert loaded.hypers["num_neighbors_adaptive"] == [6, 8]
    assert loaded.backend.num_neighbors_adaptive == 6.0

    # but other hypers still can't change
    other = _hypers([6, 8])
    other["d_pet"] += 1
    with pytest.raises(ValueError):
        model.restart(DATASET_INFO, other)

    # and the adaptive cutoff can't be switched on or off
    with pytest.raises(ValueError):
        model.set_num_neighbors_adaptive(None)
    with pytest.raises(ValueError):
        model.restart(DATASET_INFO, _hypers(None))
    no_adaptive = PET(_hypers(None), DATASET_INFO)
    with pytest.raises(ValueError):
        no_adaptive.set_num_neighbors_adaptive(8)


def test_torchscript():
    torch.manual_seed(0)
    model = PET(_hypers([4, 12]), DATASET_INFO)
    system = _system(model)
    outputs = {"energy": ModelOutput(sample_kind="system")}

    scripted = torch.jit.script(model)
    scripted.train()
    scripted([system], outputs)

    model.eval()
    scripted.eval()
    torch.testing.assert_close(
        model([system], outputs)["energy"].block().values,
        scripted([system], outputs)["energy"].block().values,
    )


class TestVariableAdaptiveTraining(PETTests):
    def test_train(self, monkeypatch, tmp_path, dataset_path, dataset_targets):
        """Training samples the target from the range, validation uses ``low``."""
        monkeypatch.chdir(tmp_path)
        dataset, _, dataset_info = self.get_dataset(dataset_targets, dataset_path)

        hypers = copy.deepcopy(DEFAULT_HYPERS)
        hypers["training"]["num_epochs"] = 1
        hypers["training"]["batch_size"] = 5
        loss_conf = OmegaConf.create(
            {k: init_with_defaults(LossSpecification) for k in dataset_targets}
        )
        OmegaConf.resolve(loss_conf)
        hypers["training"]["loss"] = loss_conf

        model_hypers = copy.deepcopy(hypers["model"])
        model_hypers["num_neighbors_adaptive"] = [3, 8]

        trainer = Trainer(hypers["training"])
        model = trainer.setup(model_hypers, dataset_info)

        # record (is validation, target) for each forward pass
        recorded = []
        original = backend_module.compute_batch_tensors

        def wrapper(*args, **kwargs):
            is_validation = not model.core.backend.training
            recorded.append((is_validation, args[10]))
            return original(*args, **kwargs)

        monkeypatch.setattr(backend_module, "compute_batch_tensors", wrapper)

        trainer.train(
            model=model,
            dtype=torch.float32,
            devices=[torch.device("cpu")],
            train_datasets=[dataset],
            val_datasets=[dataset],
            checkpoint_dir=".",
        )

        train_targets = [t for is_val, t in recorded if not is_val]
        val_targets = [t for is_val, t in recorded if is_val]
        assert len(train_targets) > 0 and len(val_targets) > 0
        assert all(3 <= t <= 8 for t in train_targets)
        assert len(set(train_targets)) > 1
        assert set(val_targets) == {3.0}
        assert model.core.backend.training
