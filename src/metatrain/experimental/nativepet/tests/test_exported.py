import pytest
import torch
from metatensor.torch.atomistic import ModelEvaluationOptions, ModelMetadata, System

from metatrain.experimental.nativepet import NativePET
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)

from . import MODEL_HYPERS


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_to(device, dtype):
    """Tests that the `.to()` method of the exported model works."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "energy": get_energy_target_info({"quantity": "energy", "unit": "eV"})
        },
    )
    model = NativePET(MODEL_HYPERS, dataset_info).to(dtype=dtype)

    exported = model.export(metadata=ModelMetadata(name="test"))

    # test correct metadata
    assert "This is the test model" in str(exported.metadata())

    exported.to(device=device)

    system = System(
        types=torch.tensor([6, 6]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )
    requested_neighbor_lists = get_requested_neighbor_lists(exported)
    system = get_system_with_neighbor_lists(system, requested_neighbor_lists)
    system = system.to(device=device, dtype=dtype)

    evaluation_options = ModelEvaluationOptions(
        length_unit=dataset_info.length_unit,
        outputs=model.outputs,
    )

    exported([system], evaluation_options, check_consistency=True)
