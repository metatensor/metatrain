import pytest
import torch
from metatensor.torch.atomistic import (
    ModelCapabilities,
    ModelEvaluationOptions,
    ModelOutput,
    System,
)
from pet.hypers import Hypers
from pet.pet import PET

from metatrain.experimental.pet import PET as WrappedPET
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import DatasetInfo, TargetInfo, TargetInfoDict
from metatrain.utils.export import export
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists


DEFAULT_HYPERS = get_default_hypers("experimental.pet")


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_to(device):
    """Tests that the `.to()` method of the exported model works."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    dtype = torch.float32  # for now
    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types={1, 6, 7, 8},
        targets=TargetInfoDict(energy=TargetInfo(quantity="energy", unit="eV")),
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

    exported = export(model, capabilities)
    exported.to(device=device, dtype=dtype)

    system = System(
        types=torch.tensor([6, 6]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        cell=torch.zeros(3, 3),
    )
    system = get_system_with_neighbor_lists(system, exported.requested_neighbor_lists())
    system = system.to(device=device, dtype=dtype)

    evaluation_options = ModelEvaluationOptions(
        length_unit=dataset_info.length_unit,
        outputs=capabilities.outputs,
    )

    exported([system], evaluation_options, check_consistency=True)
