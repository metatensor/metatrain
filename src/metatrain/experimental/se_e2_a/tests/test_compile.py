"""Tests for the tensor forward used by the scripted model."""

import torch
from metatomic.torch import ModelOutput

from metatrain.experimental.se_e2_a import SeE2A
from metatrain.experimental.se_e2_a.modules.structures import concatenate_structures
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.readers import read_systems
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from . import DATASET_PATH
from .test_basic import _minimal_hypers


def _make_model_and_systems():
    hypers = _minimal_hypers("experimental.se_e2_a")
    hypers["descriptor"]["precision"] = 64
    hypers["fitting_net"]["precision"] = 64
    targets = {
        "mtt::U0": get_energy_target_info(
            "mtt::U0", {"quantity": "energy", "unit": "eV"}
        )
    }
    dataset_info = DatasetInfo(
        length_unit="Angstrom", atomic_types=[1, 6, 7, 8], targets=targets
    )
    model = SeE2A(hypers, dataset_info)
    model.to(torch.float64)
    systems = read_systems(DATASET_PATH)[:3]
    systems = [system.to(torch.float64) for system in systems]
    for system in systems:
        get_system_with_neighbor_lists(system, model.requested_neighbor_lists())
    return model, systems


def test_forward_from_batch_matches_forward():
    """The tensor forward returns one energy per system."""
    model, systems = _make_model_and_systems()
    model.eval()
    out = model(systems, {"mtt::U0": ModelOutput(unit="", sample_kind="system")})
    assert out["mtt::U0"].block().values.shape[0] == len(systems)

    positions, species, cells, _, _ = concatenate_structures(systems)
    atype = model._prepare_atype(species)
    box = None if torch.all(cells == 0).item() else cells
    raw = model._forward_from_batch(positions, atype, box)
    assert raw["atom_energy"].numel() > 0
    assert raw["energy"].shape[0] == len(systems)
    summed = raw["atom_energy"].reshape(len(systems), -1).sum(dim=1)
    torch.testing.assert_close(summed, raw["energy"].reshape(-1))


def test_forward_from_batch_training_mode():
    """The tensor forward still returns energies while the module is training."""
    model, systems = _make_model_and_systems()
    model.train()
    positions, species, cells, _, _ = concatenate_structures(systems)
    atype = model._prepare_atype(species)
    box = None if torch.all(cells == 0).item() else cells
    raw = model._forward_from_batch(positions, atype, box)
    assert raw["atom_energy"].numel() > 0
    assert raw["energy"].shape[0] == len(systems)
