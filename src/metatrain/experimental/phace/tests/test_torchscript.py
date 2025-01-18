import pytest
import torch
from metatensor.torch.atomistic import System

from metatrain.experimental.phace import PhACE
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import (
    get_energy_target_info,
    get_generic_target_info,
)
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from . import MODEL_HYPERS


def test_torchscript():
    """Tests that the model can be jitted."""

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={"energy": get_energy_target_info({"unit": "eV"})},
    )
    model = PhACE(MODEL_HYPERS, dataset_info)
    model = torch.jit.script(model)

    system = System(
        types=torch.tensor([6, 1, 8, 7]),
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0]]
        ),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )
    get_system_with_neighbor_lists(system, model.requested_neighbor_lists())
    model(
        [system],
        {"energy": model.outputs["energy"]},
    )


@pytest.mark.parametrize("o3_lambda", [0, 1, 2, 3])
@pytest.mark.parametrize("o3_sigma", [1])
def test_torchscript_spherical(o3_lambda, o3_sigma):
    """Tests that the spherical modules can be jitted."""

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "spherical_target": get_generic_target_info(
                {
                    "quantity": "",
                    "unit": "",
                    "type": {
                        "spherical": {
                            "irreps": [{"o3_lambda": o3_lambda, "o3_sigma": o3_sigma}]
                        }
                    },
                    "num_subtargets": 100,
                    "per_atom": False,
                }
            )
        },
    )
    model = PhACE(MODEL_HYPERS, dataset_info)
    model = torch.jit.script(model)

    system = System(
        types=torch.tensor([6, 1, 8, 7]),
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0]]
        ),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )
    get_system_with_neighbor_lists(system, model.requested_neighbor_lists())
    model(
        [system],
        {"spherical_target": model.outputs["spherical_target"]},
    )


def test_torchscript_save_load():
    """Tests that the model can be jitted and saved."""

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={"energy": get_energy_target_info({"unit": "eV"})},
    )
    model = PhACE(MODEL_HYPERS, dataset_info)
    torch.jit.save(
        torch.jit.script(model),
        "model.pt",
    )
    torch.jit.load("model.pt")
