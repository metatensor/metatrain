import pytest
import torch
from metatensor.torch.atomistic import systems_to_torch

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


@pytest.mark.parametrize("calculator", ["ewald", "pme", "p3m"])
def test_long_range(calculator):
    """Tests that the long-range module can predict successfully."""

    structures = read(RESOURCES_PATH / "carbon_reduced_100.xyz", ":10")
    systems = systems_to_torch(structures)

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 8],
        targets={"energy": get_energy_target_info({"unit": "eV"})},
    )

    hypers = get_default_hypers("soap_bpnn")
    hypers["model"]["long_range"]["enable"] = True
    hypers["model"]["long_range"]["calculator"] = calculator
    model = SoapBpnn(hypers["model"], dataset_info)
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
