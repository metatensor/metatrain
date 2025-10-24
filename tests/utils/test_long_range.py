import pytest
import torch
from metatomic.torch import systems_to_torch

from metatrain.utils.architectures import get_default_hypers, import_architecture
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.readers.ase import read
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)

from . import RESOURCES_PATH


@pytest.mark.parametrize("periodicity", [True, False])
@pytest.mark.parametrize("architecture_name", ["pet", "soap_bpnn"])
def test_long_range(periodicity, architecture_name, tmpdir):
    """Tests that the long-range module can predict successfully."""

    if periodicity:
        filename = "carbon_reduced_100.xyz"
    else:
        filename = "ethanol_reduced_100.xyz"

    structures = read(RESOURCES_PATH / filename, ":10")
    systems = systems_to_torch(structures)

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 8],
        targets={"energy": get_energy_target_info("energy", {"unit": "eV"})},
    )

    hypers = get_default_hypers(architecture_name)
    hypers["model"]["long_range"]["enable"] = True
    hypers["model"]["long_range"]["use_ewald"] = True

    architecture = import_architecture(architecture_name)
    Model = architecture.__model__
    model = Model(hypers["model"], dataset_info)
    requested_nls = get_requested_neighbor_lists(model)

    systems = [
        get_system_with_neighbor_lists(system, requested_nls) for system in systems
    ]

    model(systems, {"energy": model.outputs["energy"]})

    # now torchscripted
    model = torch.jit.script(model)
    model(systems, {"energy": model.outputs["energy"]})

    # torch.jit.save and torch.jit.load
    with tmpdir.as_cwd():
        torch.jit.save(model, "model.pt")
        model = torch.jit.load("model.pt")
        model(systems, {"energy": model.outputs["energy"]})
