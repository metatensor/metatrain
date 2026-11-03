import copy

import pytest
import torch
from metatomic.torch import ModelOutput, System

from metatrain.pet import PET
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from . import MODEL_HYPERS


CUTOFF = 5.0
SPACING = 4.0  # SPACING < CUTOFF < 2 * SPACING: the neighbor graph is a path
N_ATOMS = 10


def _energy_of_first_atom(model, positions, types):
    system = System(
        types=types,
        positions=positions,
        cell=torch.zeros(3, 3, dtype=positions.dtype),
        pbc=torch.tensor([False, False, False]),
    )
    system = get_system_with_neighbor_lists(system, model.requested_neighbor_lists())
    outputs = model([system], {"energy": ModelOutput(sample_kind="atom")})
    return outputs["energy"].block().values[0, 0]


@pytest.mark.parametrize("featurizer_type", ["feedforward", "residual"])
@pytest.mark.parametrize("num_gnn_layers", [1, 2])
def test_interaction_range(featurizer_type, num_gnn_layers):
    """The exported ``interaction_range`` must be exactly the range at which the
    energy of an atom depends on the other atoms.

    On a chain of atoms where only nearest neighbors are within the cutoff, removing
    all atoms beyond ``interaction_range`` from the first atom must leave its energy
    unchanged, while also removing the farthest atom within range must change it.
    """
    torch.manual_seed(0)

    hypers = copy.deepcopy(MODEL_HYPERS)
    hypers["cutoff"] = CUTOFF
    hypers["num_gnn_layers"] = num_gnn_layers
    hypers["featurizer_type"] = featurizer_type

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6],
        targets={
            "energy": get_energy_target_info(
                "energy", {"quantity": "energy", "unit": "eV"}
            )
        },
    )
    model = PET(hypers, dataset_info).to(torch.float64).eval()
    interaction_range = model.export().capabilities().interaction_range

    if featurizer_type == "feedforward":
        assert interaction_range == (num_gnn_layers + 1) * CUTOFF
    else:
        assert interaction_range == num_gnn_layers * CUTOFF

    # slightly perturbed straight chain, so that no gradient vanishes by symmetry
    positions = torch.zeros(N_ATOMS, 3, dtype=torch.float64)
    positions[:, 0] = SPACING * torch.arange(N_ATOMS)
    positions[:, 1:] = 0.1 * torch.randn(N_ATOMS, 2, dtype=torch.float64)
    types = torch.tensor([6, 1] * (N_ATOMS // 2))

    distances = torch.linalg.norm(positions - positions[0], dim=1)
    n_within = int((distances <= interaction_range).sum())
    assert n_within < N_ATOMS

    energy_full = _energy_of_first_atom(model, positions, types)
    energy_within = _energy_of_first_atom(model, positions[:n_within], types[:n_within])
    energy_too_short = _energy_of_first_atom(
        model, positions[: n_within - 1], types[: n_within - 1]
    )

    torch.testing.assert_close(energy_within, energy_full, rtol=0.0, atol=1e-12)
    assert torch.abs(energy_too_short - energy_full) > 1e-9
