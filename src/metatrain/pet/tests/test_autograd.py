import pytest
import torch
from metatomic.torch import ModelOutput, System

from metatrain.pet import PET
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import (
    get_energy_target_info,
)
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from . import MODEL_HYPERS


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_autograd_positions(device):
    """Tests the basic functionality of the forward pass of the model."""

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    device = torch.device(device)

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "energy": get_energy_target_info({"quantity": "energy", "unit": "eV"})
        },
    )

    model = PET(MODEL_HYPERS, dataset_info)
    model = model.to(dtype=torch.float64, device=device)

    def compute(positions):
        device = positions.device

        system = System(
            types=torch.tensor([6, 6], device=device),
            positions=positions,
            cell=torch.eye(3, dtype=torch.float64, device=device),
            pbc=torch.tensor([True, True, True], device=device),
        )

        system = get_system_with_neighbor_lists(
            system, model.requested_neighbor_lists()
        )

        outputs = {"energy": ModelOutput(per_atom=False)}
        output = model([system], outputs)
        energy = output["energy"].block().values.sum()
        return energy

    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        dtype=torch.float64,
        requires_grad=True,
        device=device,
    )
    assert torch.autograd.gradcheck(compute, positions, fast_mode=True)
    assert torch.autograd.gradgradcheck(compute, positions, fast_mode=True)


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_autograd_cell(device):
    """Tests the basic functionality of the forward pass of the model."""

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    device = torch.device(device)

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "energy": get_energy_target_info({"quantity": "energy", "unit": "eV"})
        },
    )

    model = PET(MODEL_HYPERS, dataset_info)
    model = model.to(dtype=torch.float64, device=device)

    def compute(cell):
        device = cell.device

        system = System(
            types=torch.tensor([6, 6], device=device),
            positions=torch.tensor(
                [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
                dtype=torch.float64,
                device=device,
                requires_grad=True,
            ),
            cell=cell,
            pbc=torch.tensor([True, True, True], device=device),
        )

        system = get_system_with_neighbor_lists(
            system, model.requested_neighbor_lists()
        )

        outputs = {"energy": ModelOutput(per_atom=False)}
        output = model([system], outputs)
        energy = output["energy"].block().values.sum()
        return energy

    cell = torch.eye(3, dtype=torch.float64, requires_grad=True, device=device)

    assert torch.autograd.gradcheck(compute, cell, fast_mode=True)
    assert torch.autograd.gradgradcheck(compute, cell, fast_mode=True)
