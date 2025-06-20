import pytest
import torch
from metatomic.torch import systems_to_torch

from metatrain.pet import PET
from metatrain.soap_bpnn import SoapBpnn
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.readers.ase import read
from metatrain.utils.data.target_info import (
    get_energy_target_info,
)
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)

from . import RESOURCES_PATH


@pytest.mark.parametrize("periodicity", [True, False])
@pytest.mark.parametrize(
    "model_name, model_cls", [("pet", PET), ("soap_bpnn", SoapBpnn)]
)
def test_long_range(periodicity, model_name, model_cls, tmpdir):
    """Tests that the long-range module can predict successfully."""

    if periodicity:
        structures = read(RESOURCES_PATH / "carbon_reduced_100.xyz", ":10")
    else:
        structures = read(RESOURCES_PATH / "ethanol_reduced_100.xyz", ":10")
    systems = systems_to_torch(structures)

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 8],
        targets={"energy": get_energy_target_info({"unit": "eV"})},
    )

    hypers = get_default_hypers(model_name)
    hypers["model"]["long_range"]["enable"] = True
    hypers["model"]["long_range"]["use_ewald"] = True
    model = model_cls(hypers["model"], dataset_info)
    requested_nls = get_requested_neighbor_lists(model)

    systems = [
        get_system_with_neighbor_lists(system, requested_nls) for system in systems
    ]

    model(
        systems,
        {"energy": model.outputs["energy"]},
    )

    # now torchscripted
    model = torch.jit.script(model)
    model(
        systems,
        {"energy": model.outputs["energy"]},
    )

    # torch.jit.save and torch.jit.load
    with tmpdir.as_cwd():
        torch.jit.save(model, "model.pt")
        model = torch.jit.load("model.pt")
        model(
            systems,
            {"energy": model.outputs["energy"]},
        )
