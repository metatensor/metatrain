import copy
import math

import pytest


pytest.importorskip("torchpme")

import torch
from metatomic.torch import ModelOutput, System
from omegaconf import OmegaConf

from metatrain.experimental.lorem import LOREM, Trainer
from metatrain.utils.data import Dataset, DatasetInfo
from metatrain.utils.data.readers import read_systems, read_targets
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.evaluate_model import evaluate_model
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from . import DATASET_WITH_FORCES_PATH, DEFAULT_HYPERS, MODEL_HYPERS


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


def _small_lr_hypers(max_degree=1, max_degree_lr=1, use_ewald=True):
    hypers = copy.deepcopy(MODEL_HYPERS)
    hypers["max_degree"] = max_degree
    hypers["max_degree_lr"] = max_degree_lr
    hypers["num_features"] = 8
    hypers["num_spherical_features"] = 2
    hypers["num_radial"] = 4
    hypers["long_range"]["use_ewald"] = use_ewald
    return hypers


def _chain_system(pbc=True):
    cell = torch.eye(3) * 10 if pbc else torch.zeros(3, 3)
    return System(
        types=torch.tensor([6, 6, 8, 8]),
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0]]
        ),
        cell=cell,
        pbc=torch.tensor([pbc, pbc, pbc]),
    )


@pytest.mark.parametrize("use_ewald", [True, False])
def test_long_range_features(use_ewald):
    """Long-range Coulomb features can be evaluated for a tiny periodic system."""
    model = LOREM(_small_lr_hypers(use_ewald=use_ewald), _energy_dataset_info())
    system = get_system_with_neighbor_lists(
        _chain_system(), model.requested_neighbor_lists()
    )
    outputs = {"energy": ModelOutput(sample_kind="system")}
    model([system, system], outputs)


def test_sr_lr_module_scopes():
    """Top-level param scopes: backbone is ``sr``, long-range is ``lr``."""
    model = LOREM(_small_lr_hypers(), _energy_dataset_info())
    names = {name for name, _ in model.named_children()}
    assert "sr" in names
    assert "lr" in names
    assert any(key.startswith("sr.") for key in model.state_dict())
    assert any(key.startswith("lr.lr_scale") for key in model.state_dict())


def test_lr_scale_zero_is_noop():
    """``lr_scale == 0`` leaves short-range features unchanged (a zero
    perturbation, for warm-starting the short-range trunk)."""
    hypers = _small_lr_hypers()
    hypers["long_range"]["lr_scale_init"] = 0.0
    model = LOREM(hypers, _energy_dataset_info())
    model.eval()
    system = get_system_with_neighbor_lists(
        _chain_system(), model.requested_neighbor_lists()
    )
    features, distances, spherical = model.sr([system])
    updated = model.lr([system], features, distances, spherical)
    torch.testing.assert_close(updated, features, atol=1e-5, rtol=1e-5)


def test_max_degree_lr_cannot_exceed_max_degree():
    hypers = _small_lr_hypers(max_degree=1, max_degree_lr=2)
    with pytest.raises(ValueError, match="max_degree_lr"):
        LOREM(hypers, _energy_dataset_info())


def test_long_range_energy_rotation_invariant():
    """System energy is unchanged when the molecule is rotated."""
    hypers = _small_lr_hypers(max_degree=2, max_degree_lr=2, use_ewald=True)
    model = LOREM(hypers, _energy_dataset_info())
    model.eval()

    system = _chain_system(pbc=False)
    theta = math.pi / 3.0
    rotation = torch.tensor(
        [
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta), math.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotated = System(
        types=system.types,
        positions=system.positions @ rotation.T,
        cell=system.cell,
        pbc=system.pbc,
    )

    options = model.requested_neighbor_lists()
    system = get_system_with_neighbor_lists(system, options)
    rotated = get_system_with_neighbor_lists(rotated, options)
    outputs = {"energy": ModelOutput(sample_kind="system")}

    energy = model([system], outputs)["energy"].block().values
    energy_rotated = model([rotated], outputs)["energy"].block().values
    torch.testing.assert_close(energy, energy_rotated, atol=1e-5, rtol=1e-5)


def test_spherical_charges_rotate_as_vectors():
    """ℓ=1 long-range charges transform as Cartesian vectors under rotation."""
    hypers = _small_lr_hypers(max_degree=1, max_degree_lr=1, use_ewald=True)
    model = LOREM(hypers, _energy_dataset_info())
    model.eval()

    system = _chain_system(pbc=False)
    theta = math.pi / 5.0
    # Real SH ℓ=1 order is Y, Z, X. A rotation about z mixes X and Y only.
    rotation = torch.tensor(
        [
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta), math.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotated = System(
        types=system.types,
        positions=system.positions @ rotation.T,
        cell=system.cell,
        pbc=system.pbc,
    )
    options = model.requested_neighbor_lists()
    system = get_system_with_neighbor_lists(system, options)
    rotated = get_system_with_neighbor_lists(rotated, options)

    features, _, spherical = model.sr([system])
    features_rot, _, spherical_rot = model.sr([rotated])
    charges = model.lr.map_charges(features, spherical)
    charges_rot = model.lr.map_charges(features_rot, spherical_rot)

    # channels: [scalar, Y00, Y1,-1 (y), Y1,0 (z), Y1,+1 (x)]
    dipole = charges[:, 2:5]
    dipole_rot = charges_rot[:, 2:5]
    # Components are (y, z, x). Rotate the Cartesian vector (x, y, z).
    xyz = torch.stack([dipole[:, 2], dipole[:, 0], dipole[:, 1]], dim=-1)
    xyz_expected = xyz @ rotation.T
    dipole_expected = torch.stack(
        [xyz_expected[:, 1], xyz_expected[:, 2], xyz_expected[:, 0]], dim=-1
    )
    torch.testing.assert_close(dipole_rot, dipole_expected, atol=1e-5, rtol=1e-5)


def test_long_range_torchscript():
    """The equivariant long-range path is TorchScript-compilable."""
    model = LOREM(_small_lr_hypers(), _energy_dataset_info())
    scripted = torch.jit.script(model)
    system = get_system_with_neighbor_lists(
        _chain_system(), model.requested_neighbor_lists()
    )
    outputs = {"energy": ModelOutput(sample_kind="system")}
    eager = model([system], outputs)["energy"].block().values
    compiled = scripted([system], outputs)["energy"].block().values
    torch.testing.assert_close(eager, compiled, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("use_ewald", [True, False])
def test_long_range_training(use_ewald):
    """A short training run succeeds with long-range features enabled."""
    systems = read_systems(DATASET_WITH_FORCES_PATH)
    conf = {
        "energy": {
            "quantity": "energy",
            "read_from": DATASET_WITH_FORCES_PATH,
            "reader": "ase",
            "key": "energy",
            "unit": "eV",
            "type": "scalar",
            "sample_kind": "system",
            "num_subtargets": 1,
            "forces": {"read_from": DATASET_WITH_FORCES_PATH, "key": "force"},
            "stress": False,
            "virial": False,
        }
    }
    targets, target_info_dict = read_targets(OmegaConf.create(conf))
    dataset = Dataset.from_dict({"system": systems, "energy": targets["energy"]})
    hypers = copy.deepcopy(DEFAULT_HYPERS)
    hypers["training"]["num_epochs"] = 2
    hypers["training"]["scheduler_patience"] = 1
    hypers["training"]["fixed_composition_weights"] = {}

    dataset_info = DatasetInfo(
        length_unit="Angstrom", atomic_types=[6], targets=target_info_dict
    )
    model_hypers = _small_lr_hypers(use_ewald=use_ewald)
    model = LOREM(model_hypers, dataset_info)

    trainer = Trainer(hypers["training"])
    trainer.train(
        model=model,
        dtype=torch.float32,
        devices=[torch.device("cpu")],
        train_datasets=[dataset],
        val_datasets=[dataset],
        checkpoint_dir=".",
    )

    systems = [system.to(torch.float32) for system in systems]
    for system in systems:
        get_system_with_neighbor_lists(system, model.requested_neighbor_lists())
    evaluate_model(model, systems[:5], targets=target_info_dict, is_training=False)
