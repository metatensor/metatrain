import ase
import pytest
import torch
from metatensor.torch.atomistic import ModelEvaluationOptions, systems_to_torch

from metatensor.models.experimental.soap_bpnn import SOAPBPNN
from metatensor.models.utils.architectures import get_default_hypers
from metatensor.models.utils.data import DatasetInfo, TargetInfo
from metatensor.models.utils.neighbor_lists import get_system_with_neighbor_lists


DEFAULT_HYPERS = get_default_hypers("experimental.soap_bpnn")


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_to(device, dtype):
    """Tests that the `.to()` method of the exported model works."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "energy": TargetInfo(
                quantity="energy",
                unit="eV",
            )
        },
    )
    model = SOAPBPNN(DEFAULT_HYPERS["model"], dataset_info).to(dtype=dtype)
    exported = model.export()

    exported.to(device=device)

    system = ase.Atoms("O2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    system = systems_to_torch(system, dtype=torch.get_default_dtype())
    system = get_system_with_neighbor_lists(system, exported.requested_neighbor_lists())
    system = system.to(device=device, dtype=dtype)

    evaluation_options = ModelEvaluationOptions(
        length_unit=dataset_info.length_unit,
        outputs=model.outputs,
    )

    exported([system], evaluation_options, check_consistency=True)
