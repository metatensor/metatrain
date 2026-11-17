"""Tests for ``num_neighbors_adaptive`` given as a ``[low, high]`` range."""

import copy
import math

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


def _system(model, n_atoms=12, seed=1):
    torch.manual_seed(seed)
    system = System(
        types=torch.full((n_atoms,), 6),
        positions=torch.rand(n_atoms, 3) * 5.0,
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )
    return get_system_with_neighbor_lists(system, model.requested_neighbor_lists())


def _per_system(targets, sizes):
    """Split per-atom targets into per-system values, checking that each system
    uses a single target."""
    values = []
    for chunk in torch.split(targets, sizes):
        assert torch.all(chunk == chunk[0])
        values.append(chunk[0].item())
    return values


@pytest.fixture
def recorded_targets(monkeypatch):
    """Record the per-atom target number of neighbors passed to
    ``compute_batch_tensors``."""
    recorded = []
    original = backend_module.compute_batch_tensors

    def wrapper(*args, **kwargs):
        recorded.append(args[10].detach().clone())
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


def _sample(low, high, n):
    return sample_num_neighbors_adaptive(
        low, high, n, torch.device("cpu"), torch.float64
    )


def test_sample():
    torch.manual_seed(0)
    samples = _sample(4, 7, 200)
    assert samples.shape == (200,)
    assert set(samples.tolist()) == {4.0, 5.0, 6.0, 7.0}

    # the samples follow the discretized log-uniform distribution
    low, high = 4, 64
    samples = _sample(low, high, 20000)
    assert samples.min() == low and samples.max() == high
    for k in [4, 8, 16, 32]:
        expected = math.log((high + 1) / k) / math.log((high + 1) / low)
        observed = (samples >= k).double().mean().item()
        assert abs(observed - expected) < 0.02

    # a degenerate range does not consume random numbers
    state = torch.get_rng_state()
    assert torch.equal(_sample(5, 5, 3), torch.full((3,), 5.0, dtype=torch.float64))
    assert torch.equal(state, torch.get_rng_state())


def test_training_samples_per_structure(recorded_targets):
    torch.manual_seed(0)
    model = PET(_hypers([4, 12]), DATASET_INFO)
    sizes = [12, 7, 9]
    systems = [_system(model, n, seed=i) for i, n in enumerate(sizes)]
    outputs = {"energy": ModelOutput(sample_kind="system")}

    model.train()
    for _ in range(20):
        model(systems, outputs)
    per_system = [_per_system(t, sizes) for t in recorded_targets]
    values = [v for batch in per_system for v in batch]
    assert all(v == int(v) and 4 <= v <= 12 for v in values)
    # structures in the same batch get different targets
    assert any(len(set(batch)) > 1 for batch in per_system)

    recorded_targets.clear()
    model.eval()
    model(systems, outputs)
    assert _per_system(recorded_targets[0], sizes) == [4.0, 4.0, 4.0]


def test_per_atom_targets_match_single_structures(recorded_targets):
    """Atoms of a structure only see the target sampled for that structure: the
    cutoffs in a batch with targets (a, b) match the ones of each structure
    evaluated alone with a fixed target."""
    torch.manual_seed(0)
    model = PET(_hypers([4, 12]), DATASET_INFO)
    sizes = [12, 9]
    systems = [_system(model, n, seed=i) for i, n in enumerate(sizes)]
    outputs = {"mtt::aux::cutoff_stats": ModelOutput(sample_kind="atom")}

    model.train()
    # run until the two structures get different targets
    for _ in range(20):
        recorded_targets.clear()
        batch = model(systems, outputs)["mtt::aux::cutoff_stats"].block()
        targets = _per_system(recorded_targets[0], sizes)
        if targets[0] != targets[1]:
            break
    assert targets[0] != targets[1]

    for i, (system, target) in enumerate(zip(systems, targets, strict=True)):
        model.set_num_neighbors_adaptive(int(target))
        single = model([system], outputs)["mtt::aux::cutoff_stats"].block()
        mask = batch.samples.column("system") == i
        torch.testing.assert_close(batch.values[mask], single.values)


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
    assert set(recorded_targets[0].tolist()) == {10.0}

    # restarting with a different value is allowed
    recorded_targets.clear()
    model.restart(DATASET_INFO, _hypers([6, 8]))
    assert model.hypers["num_neighbors_adaptive"] == [6, 8]
    model([system], outputs)
    model.train()
    for _ in range(20):
        model([system], outputs)
    assert set(recorded_targets[0].tolist()) == {6.0}
    assert set(torch.cat(recorded_targets[1:]).tolist()) <= {6.0, 7.0, 8.0}
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
        """Training samples the targets from the range, validation uses ``low``."""
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
            recorded.append((is_validation, args[10].detach().clone()))
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
        assert all(t.min() >= 3 and t.max() <= 8 for t in train_targets)
        # several targets are used inside the same batch
        assert any(len(t.unique()) > 1 for t in train_targets)
        assert set(torch.cat(val_targets).tolist()) == {3.0}
        assert model.core.backend.training
