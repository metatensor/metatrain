import os

import ase
import pytest
import torch
from metatensor.torch.atomistic import (
    ModelCapabilities,
    ModelEvaluationOptions,
    ModelOutput,
    systems_to_torch,
)

from metatensor.models.experimental.pet import DEFAULT_HYPERS, Model
from metatensor.models.utils.io import export, load
from metatensor.models.utils.neighbors_lists import get_system_with_neighbors_lists


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_to(tmp_path, device, dtype):
    """Tests that the `.to()` method of the exported model works."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    os.chdir(tmp_path)
    dtype = torch.float32  # for now
    capabilities = ModelCapabilities(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        outputs={
            "energy": ModelOutput(
                quantity="energy",
                unit="eV",
            )
        },
    )
    pet = Model(capabilities, DEFAULT_HYPERS["ARCHITECTURAL_HYPERS"])
    export(pet, "pet.pt")
    exported = load("pet.pt")

    exported.to(device=device, dtype=dtype)

    system = ase.Atoms("O2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    system = systems_to_torch(system, dtype=torch.get_default_dtype())
    system = get_system_with_neighbors_lists(
        system, exported.requested_neighbors_lists()
    )
    system = system.to(device=device, dtype=dtype)

    evaluation_options = ModelEvaluationOptions(
        length_unit=capabilities.length_unit,
        outputs=capabilities.outputs,
    )

    exported([system], evaluation_options, check_consistency=True)
