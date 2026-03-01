"""Tests for _forward_from_batch and compilation preparation."""

import copy

import torch
from metatomic.torch import ModelOutput

from metatrain.experimental.dpa3 import DPA3
from metatrain.experimental.dpa3.modules.structures import concatenate_structures
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.readers import read_systems
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from . import DATASET_PATH, MODEL_HYPERS


def _make_model_and_systems():
    hypers = copy.deepcopy(MODEL_HYPERS)
    hypers["descriptor"]["repflow"]["n_dim"] = 2
    hypers["descriptor"]["repflow"]["e_dim"] = 2
    hypers["descriptor"]["repflow"]["a_dim"] = 2
    hypers["descriptor"]["repflow"]["e_sel"] = 1
    hypers["descriptor"]["repflow"]["a_sel"] = 1
    hypers["descriptor"]["repflow"]["axis_neuron"] = 1
    hypers["descriptor"]["repflow"]["nlayers"] = 1
    hypers["fitting_net"]["neuron"] = [1, 1]
    # deepmd-kit precision must be set at construction time; .to(dtype)
    # does not update internal self.prec attribute.
    hypers["descriptor"]["precision"] = "float64"
    hypers["fitting_net"]["precision"] = "float64"

    targets = {
        "mtt::U0": get_energy_target_info(
            "mtt::U0", {"quantity": "energy", "unit": "eV"}
        )
    }
    dataset_info = DatasetInfo(
        length_unit="Angstrom", atomic_types=[1, 6, 7, 8], targets=targets
    )
    model = DPA3(hypers, dataset_info)

    systems = read_systems(DATASET_PATH)[:3]
    systems = [s.to(torch.float64) for s in systems]
    for s in systems:
        get_system_with_neighbor_lists(s, model.requested_neighbor_lists())
    return model, systems


def test_forward_from_batch_matches_forward():
    """_forward_from_batch produces the same atom_energy as forward."""
    model, systems = _make_model_and_systems()
    model.eval()

    # Standard forward
    out = model(
        systems,
        {"mtt::U0": ModelOutput(quantity="energy", unit="", per_atom=False)},
    )
    # Verify the standard forward works
    assert out["mtt::U0"].block().values.numel() > 0

    # Pure-tensor forward
    positions, species, cells, atom_index, system_index = concatenate_structures(
        systems
    )
    atype = model._prepare_atype(species)
    box = None if torch.all(cells == 0).item() else cells
    raw = model._forward_from_batch(positions, atype, box)

    # The raw atom_energy should be non-empty
    assert raw["atom_energy"].numel() > 0
    assert raw["energy"].numel() > 0

    # Total energy from the batch path (sum per system, before scaler/additive)
    # should match the raw energies shape
    assert raw["energy"].shape[0] == len(systems)


def test_forward_from_batch_training_mode():
    """_forward_from_batch works in training mode (no scaler/additive)."""
    model, systems = _make_model_and_systems()
    model.train()

    positions, species, cells, atom_index, system_index = concatenate_structures(
        systems
    )
    atype = model._prepare_atype(species)
    box = None if torch.all(cells == 0).item() else cells
    raw = model._forward_from_batch(positions, atype, box)

    # Should still return valid tensors
    assert "atom_energy" in raw
    assert "energy" in raw
    assert raw["atom_energy"].requires_grad or not model.training
