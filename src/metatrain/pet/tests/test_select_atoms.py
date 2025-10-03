import copy

import pytest
import torch
from metatensor.torch import Labels

from metatrain.pet import PET
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.readers import (
    read_systems,
)
from metatrain.utils.data.target_info import (
    get_energy_target_info,
)
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from . import DATASET_PATH, MODEL_HYPERS


@pytest.mark.parametrize("select_atoms", [[0, 2]])
def test_select_atoms(select_atoms):
    """Tests that features are returned only for selected atoms."""

    systems = read_systems(DATASET_PATH)
    systems = [system.to(torch.float32) for system in systems]

    target_info_dict = {}
    target_info_dict["energy"] = get_energy_target_info(
        {"quantity": "energy", "unit": "eV"}
    )
    dataset_info = DatasetInfo(
        length_unit="Angstrom", atomic_types=[1, 6, 7, 8], targets=target_info_dict
    )

    hypers = copy.deepcopy(MODEL_HYPERS)
    model = PET(hypers, dataset_info)
    systems = [
        get_system_with_neighbor_lists(system, model.requested_neighbor_lists())
        for system in systems
    ]

    output_label = "mtt::aux::energy_last_layer_features"
    modeloutput = model.outputs[output_label]
    modeloutput.per_atom = True
    outputs = {output_label: modeloutput}
    selected_atoms = Labels(
        names=["system", "atom"],
        values=torch.tensor(
            [(n, i) for n in range(len(systems)) for i in select_atoms]
        ),
    )
    out = model(systems, outputs, selected_atoms=selected_atoms)
    features = out[output_label].block().samples.values
    assert features.shape == selected_atoms.values.shape
