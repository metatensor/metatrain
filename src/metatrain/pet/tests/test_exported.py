import pytest
import torch
from metatensor.torch.atomistic import (
    ModelCapabilities,
    ModelEvaluationOptions,
    ModelMetadata,
    ModelOutput,
    System,
)

from metatrain.pet import PET as WrappedPET
from metatrain.pet.modules.hypers import Hypers
from metatrain.pet.modules.pet import PET
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)


DEFAULT_HYPERS = get_default_hypers("pet")


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_to(device):
    """Tests that the `.to()` method of the exported model works."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    dtype = torch.float32  # for now
    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={"energy": get_energy_target_info({"unit": "eV"})},
    )
    model = WrappedPET(DEFAULT_HYPERS["model"], dataset_info)
    ARCHITECTURAL_HYPERS = Hypers(model.hypers)
    raw_pet = PET(ARCHITECTURAL_HYPERS, 0.0, len(model.atomic_types))
    model.set_trained_model(raw_pet)

    capabilities = ModelCapabilities(
        length_unit="Angstrom",
        atomic_types=model.atomic_types,
        outputs={
            "energy": ModelOutput(
                quantity="energy",
                unit="eV",
            )
        },
        interaction_range=DEFAULT_HYPERS["model"]["N_GNN_LAYERS"]
        * DEFAULT_HYPERS["model"]["R_CUT"],
        dtype="float32",
        supported_devices=["cpu", "cuda"],
    )

    exported = model.export(metadata=ModelMetadata(name="test"))

    # test correct metadata
    assert "This is the test model" in str(exported.metadata())

    exported.to(device=device, dtype=dtype)

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
        outputs=capabilities.outputs,
    )

    exported([system], evaluation_options, check_consistency=True)
