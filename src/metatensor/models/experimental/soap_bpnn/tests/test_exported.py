import ase
import pytest
import torch
from metatensor.torch.atomistic import (
    ModelCapabilities,
    ModelEvaluationOptions,
    ModelOutput,
    systems_to_torch,
)

from metatensor.models.experimental.soap_bpnn import DEFAULT_HYPERS, Model
from metatensor.models.utils.export import export
from metatensor.models.utils.neighbor_lists import get_system_with_neighbor_lists


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_to(device, dtype):
    """Tests that the `.to()` method of the exported model works."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    if dtype == torch.float32:
        dtype_string = "float32"
    elif dtype == torch.float64:
        dtype_string = "float64"
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    capabilities = ModelCapabilities(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        outputs={
            "energy": ModelOutput(
                quantity="energy",
                unit="eV",
            )
        },
        interaction_range=DEFAULT_HYPERS["model"]["soap"]["cutoff"],
        dtype=dtype_string,
    )
    model = Model(capabilities, DEFAULT_HYPERS["model"]).to(dtype=dtype)
    exported = export(model)

    exported.to(device=device)

    system = ase.Atoms("O2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    system = systems_to_torch(system, dtype=torch.get_default_dtype())
    system = get_system_with_neighbor_lists(system, exported.requested_neighbor_lists())
    system = system.to(device=device, dtype=dtype)

    evaluation_options = ModelEvaluationOptions(
        length_unit=capabilities.length_unit,
        outputs=capabilities.outputs,
    )

    exported([system], evaluation_options, check_consistency=True)
