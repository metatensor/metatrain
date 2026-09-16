"""One training step stays finite for each supported target mix and batch size.

Stress is the energy ``strain`` gradient. Those setups use a periodic cell
(zero-cell systems cannot carry stress).
"""

import copy

import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import System
from omegaconf import OmegaConf

from metatrain.experimental.lorem import LOREM, Trainer
from metatrain.utils.data import Dataset, DatasetInfo
from metatrain.utils.data.target_info import (
    get_energy_target_info,
    get_generic_target_info,
)
from metatrain.utils.evaluate_model import evaluate_model
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from . import DEFAULT_HYPERS, MODEL_HYPERS


SETUPS = (
    "energy",
    "energy_forces",
    "energy_stress",
    "energy_forces_stress",
    "dipole",
    "energy_dipole",
    "energy_forces_dipole",
    "bec",
    "energy_bec",
    "energy_forces_bec",
    "energy_forces_stress_bec",
)
N_SYSTEMS = 4
N_ATOMS = 4
CELL = 8.0


def _toy_systems(n_systems: int = N_SYSTEMS, periodic: bool = False) -> list[System]:
    systems = []
    for i in range(n_systems):
        offset = 0.15 * float(i)
        positions = torch.tensor(
            [
                [offset, 0.0, 0.0],
                [offset, 0.0, 1.1],
                [offset + 0.8, 0.7, 0.2],
                [offset - 0.6, 0.5, 1.3],
            ],
            dtype=torch.float64,
        )
        if periodic:
            cell = torch.eye(3, dtype=torch.float64) * CELL
            pbc = torch.tensor([True, True, True])
        else:
            cell = torch.zeros(3, 3, dtype=torch.float64)
            pbc = torch.tensor([False, False, False])
        systems.append(
            System(
                types=torch.tensor([6, 6, 1, 1]),
                positions=positions,
                cell=cell,
                pbc=pbc,
            )
        )
    return systems


def _energy_maps(
    systems: list[System], with_forces: bool, with_stress: bool
) -> list[TensorMap]:
    maps = []
    for index, system in enumerate(systems):
        n_atoms = len(system.types)
        block = TensorBlock(
            values=torch.tensor([[-10.0 - 0.1 * index]], dtype=torch.float64),
            samples=Labels(["system"], torch.tensor([[index]], dtype=torch.int32)),
            components=[],
            properties=Labels("energy", torch.tensor([[0]])),
        )
        if with_forces:
            gradient = TensorBlock(
                values=0.05 * torch.randn(n_atoms, 3, 1, dtype=torch.float64),
                samples=Labels(
                    ["sample", "system", "atom"],
                    torch.tensor(
                        [[0, index, atom] for atom in range(n_atoms)],
                        dtype=torch.int32,
                    ),
                ),
                components=[Labels(["xyz"], torch.arange(3).reshape(-1, 1))],
                properties=Labels("energy", torch.tensor([[0]])),
            )
            block.add_gradient("positions", gradient)
        if with_stress:
            # Same layout as the ASE reader: stress × volume is the strain gradient.
            volume = float(torch.det(system.cell))
            stress = 0.02 * torch.randn(1, 3, 3, 1, dtype=torch.float64)
            stress = 0.5 * (stress + stress.transpose(1, 2))
            strain = TensorBlock(
                values=stress * volume,
                samples=Labels(["sample"], torch.tensor([[0]], dtype=torch.int32)),
                components=[
                    Labels(["xyz_1"], torch.arange(3).reshape(-1, 1)),
                    Labels(["xyz_2"], torch.arange(3).reshape(-1, 1)),
                ],
                properties=Labels("energy", torch.tensor([[0]])),
            )
            block.add_gradient("strain", strain)
        maps.append(TensorMap(Labels(["_"], torch.tensor([[0]])), [block]))
    return maps


def _bec_maps(systems: list[System]) -> list[TensorMap]:
    maps = []
    for index, system in enumerate(systems):
        n_atoms = len(system.types)
        values = torch.randn(n_atoms, 3, 3, 1, dtype=torch.float64)
        values = values - values.mean(dim=0, keepdim=True)
        block = TensorBlock(
            values=values,
            samples=Labels(
                ["system", "atom"],
                torch.stack(
                    [
                        torch.full((n_atoms,), index, dtype=torch.int32),
                        torch.arange(n_atoms, dtype=torch.int32),
                    ],
                    dim=1,
                ),
            ),
            components=[
                Labels("xyz_1", torch.arange(3).reshape(-1, 1)),
                Labels("xyz_2", torch.arange(3).reshape(-1, 1)),
            ],
            properties=Labels.range("bec", 1),
        )
        maps.append(TensorMap(Labels.single(), [block]))
    return maps


def _dipole_maps(systems: list[System]) -> list[TensorMap]:
    maps = []
    for index, _system in enumerate(systems):
        block = TensorBlock(
            values=0.1 * torch.randn(1, 3, 1, dtype=torch.float64),
            samples=Labels(["system"], torch.tensor([[index]], dtype=torch.int32)),
            components=[Labels(["xyz"], torch.arange(3).reshape(-1, 1))],
            properties=Labels.range("dipole", 1),
        )
        maps.append(TensorMap(Labels.single(), [block]))
    return maps


def _dipole_target_info():
    return get_generic_target_info(
        "dipole",
        OmegaConf.create(
            {
                "quantity": "dipole",
                "unit": "e*A",
                "type": {"cartesian": {"rank": 1}},
                "sample_kind": "system",
                "num_subtargets": 1,
            }
        ),
    )


def _bec_target_info():
    return get_generic_target_info(
        "bec",
        OmegaConf.create(
            {
                "quantity": "bec",
                "unit": "e",
                "type": {"cartesian": {"rank": 2}},
                "sample_kind": "atom",
                "num_subtargets": 1,
            }
        ),
    )


def _energy_loss(with_forces: bool, with_stress: bool) -> dict:
    gradients = {}
    if with_forces:
        gradients["positions"] = {
            "type": "mse",
            "weight": 1.0,
            "reduction": "mean",
        }
    if with_stress:
        gradients["strain"] = {
            "type": "mse",
            "weight": 1.0,
            "reduction": "mean",
        }
    return {
        "type": "mse",
        "weight": 1.0,
        "reduction": "mean",
        "gradients": gradients,
    }


def _setup_payload(setup: str):
    want_energy = setup == "energy" or setup.startswith("energy_")
    want_forces = "forces" in setup
    want_stress = "stress" in setup
    want_bec = "bec" in setup
    want_dipole = "dipole" in setup
    systems = _toy_systems(periodic=want_stress)
    payload: dict = {"system": systems}
    targets = {}
    if want_energy:
        payload["energy"] = _energy_maps(
            systems, with_forces=want_forces, with_stress=want_stress
        )
        targets["energy"] = get_energy_target_info(
            "energy",
            {"quantity": "energy", "unit": "eV"},
            add_position_gradients=want_forces,
            add_strain_gradients=want_stress,
        )
    if want_dipole:
        payload["dipole"] = _dipole_maps(systems)
        targets["dipole"] = _dipole_target_info()
    if want_bec:
        payload["bec"] = _bec_maps(systems)
        targets["bec"] = _bec_target_info()
    return payload, targets


def _small_hypers(need_bec: bool) -> dict:
    hypers = copy.deepcopy(MODEL_HYPERS)
    hypers["cutoff"] = 3.0
    hypers["cutoff_width"] = 0.5
    hypers["max_degree"] = 2 if need_bec else 1
    hypers["max_degree_lr"] = 1 if need_bec else 0
    hypers["num_features"] = 8
    hypers["num_spherical_features"] = 2
    hypers["num_radial"] = 4
    hypers["num_message_passing"] = 0
    return hypers


def _assert_finite_module(module: torch.nn.Module) -> None:
    for name, tensor in module.named_parameters():
        assert torch.isfinite(tensor).all(), f"non-finite parameter {name}"
    for name, tensor in module.named_buffers():
        if tensor.is_floating_point():
            assert torch.isfinite(tensor).all(), f"non-finite buffer {name}"


def _assert_finite_predictions(predictions: dict) -> None:
    for name, tensor_map in predictions.items():
        for block in tensor_map.blocks():
            assert torch.isfinite(block.values).all(), f"non-finite prediction {name}"
            for gradient_name, gradient in block.gradients():
                assert torch.isfinite(gradient.values).all(), (
                    f"non-finite prediction {name} {gradient_name} gradient"
                )


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("setup", SETUPS)
def test_one_training_step_is_finite(setup, batch_size, tmp_path, monkeypatch):
    """One ``Trainer`` epoch: loss, weights, and a follow-up forward stay finite."""
    pytest.importorskip("torchpme")
    monkeypatch.chdir(tmp_path)

    payload, targets = _setup_payload(setup)
    dataset = Dataset.from_dict(payload)
    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6],
        targets=targets,
    )
    model = LOREM(_small_hypers(need_bec="bec" in setup), dataset_info)

    training = copy.deepcopy(DEFAULT_HYPERS["training"])
    training["num_epochs"] = 1
    training["batch_size"] = batch_size
    training["scheduler_patience"] = 1
    training["checkpoint_interval"] = 100
    training["fixed_composition_weights"] = {}
    if "forces" in setup or "stress" in setup or "dipole" in setup:
        training["loss"] = {}
        if setup == "energy" or setup.startswith("energy_"):
            training["loss"]["energy"] = _energy_loss(
                with_forces="forces" in setup, with_stress="stress" in setup
            )
        if "dipole" in setup:
            training["loss"]["dipole"] = {
                "type": "mse",
                "weight": 1.0,
                "reduction": "mean",
                "gradients": {},
            }
        if "bec" in setup:
            training["loss"]["bec"] = {
                "type": "mse",
                "weight": 1.0,
                "reduction": "mean",
                "gradients": {},
            }

    trainer = Trainer(training)
    trainer.train(
        model=model,
        dtype=torch.float32,
        devices=[torch.device("cpu")],
        train_datasets=[dataset],
        val_datasets=[dataset],
        checkpoint_dir=".",
    )

    _assert_finite_module(model)
    assert trainer.best_metric is not None
    assert trainer.best_metric != float("inf")
    assert trainer.best_metric == trainer.best_metric  # not NaN

    systems = [
        get_system_with_neighbor_lists(
            system.to(torch.float32), model.requested_neighbor_lists()
        )
        for system in payload["system"][:2]
    ]
    predictions = evaluate_model(model, systems, targets, is_training=False)
    _assert_finite_predictions(predictions)


def test_force_step_with_isolated_atom_is_finite(tmp_path, monkeypatch):
    """Dissociated pairs (empty neighbor list) must not NaN a force step."""
    pytest.importorskip("torchpme")
    monkeypatch.chdir(tmp_path)

    compact = _toy_systems(n_systems=1, periodic=False)[0]
    isolated = System(
        types=torch.tensor([17, 17]),
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [12.0, 0.0, 0.0]], dtype=torch.float64
        ),
        cell=torch.zeros(3, 3, dtype=torch.float64),
        pbc=torch.tensor([False, False, False]),
    )
    systems = [compact, isolated]
    payload = {
        "system": systems,
        "energy": _energy_maps(systems, with_forces=True, with_stress=False),
    }
    targets = {
        "energy": get_energy_target_info(
            "energy",
            {"quantity": "energy", "unit": "eV"},
            add_position_gradients=True,
        )
    }
    model = LOREM(
        _small_hypers(need_bec=False),
        DatasetInfo(length_unit="Angstrom", atomic_types=[1, 6, 17], targets=targets),
    )
    training = copy.deepcopy(DEFAULT_HYPERS["training"])
    training["num_epochs"] = 1
    training["batch_size"] = 2
    training["scheduler_patience"] = 1
    training["checkpoint_interval"] = 100
    training["fixed_composition_weights"] = {}
    training["loss"] = {
        "energy": _energy_loss(with_forces=True, with_stress=False),
    }
    Trainer(training).train(
        model=model,
        dtype=torch.float32,
        devices=[torch.device("cpu")],
        train_datasets=[Dataset.from_dict(payload)],
        val_datasets=[Dataset.from_dict(payload)],
        checkpoint_dir=".",
    )
    _assert_finite_module(model)
