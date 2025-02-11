import copy

import torch
from metatensor.torch.atomistic import System

from metatrain.experimental.nanopet import NanoPET
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from . import MODEL_HYPERS


def test_torchscript():
    """Tests that the model can be jitted."""

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "energy": get_energy_target_info({"quantity": "energy", "unit": "eV"})
        },
    )
    model = NanoPET(MODEL_HYPERS, dataset_info)

    system = System(
        types=torch.tensor([6, 1, 8, 7]),
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0]]
        ),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )
    system = get_system_with_neighbor_lists(system, model.requested_neighbor_lists())

    model = torch.jit.script(model)
    model(
        [system],
        {"energy": model.outputs["energy"]},
    )


def test_torchscript_save_load(tmpdir):
    """Tests that the model can be jitted and saved."""

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "energy": get_energy_target_info({"quantity": "energy", "unit": "eV"})
        },
    )
    model = NanoPET(MODEL_HYPERS, dataset_info)

    with tmpdir.as_cwd():
        torch.jit.save(torch.jit.script(model), "model.pt")
        torch.jit.load("model.pt")


def test_torchscript_integers():
    """Tests that the model can be jitted when some float
    parameters are instead supplied as integers."""

    new_hypers = copy.deepcopy(MODEL_HYPERS)
    new_hypers["cutoff"] = 5
    new_hypers["cutoff_width"] = 1

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "energy": get_energy_target_info({"quantity": "energy", "unit": "eV"})
        },
    )
    model = NanoPET(new_hypers, dataset_info)

    system = System(
        types=torch.tensor([6, 1, 8, 7]),
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0]]
        ),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )
    system = get_system_with_neighbor_lists(system, model.requested_neighbor_lists())

    model = torch.jit.script(model)
    model(
        [system],
        {"energy": model.outputs["energy"]},
    )
