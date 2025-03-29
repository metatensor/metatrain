import torch
from metatensor.torch.atomistic import ModelOutput, System

from metatrain.experimental.nativepet import NativePET
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import (
    get_energy_target_info,
)
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from . import MODEL_HYPERS


def test_autograd_positions():
    """Tests the basic functionality of the forward pass of the model."""
    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "energy": get_energy_target_info({"quantity": "energy", "unit": "eV"})
        },
    )

    model = NativePET(MODEL_HYPERS, dataset_info)
    model = model.to(torch.float64)

    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=torch.float64, requires_grad=True
    )

    def compute(positions):
        system = System(
            types=torch.tensor([6, 6]),
            positions=positions,
            cell=torch.eye(3, dtype=torch.float64),
            pbc=torch.tensor([True, True, True]),
        )

        system = get_system_with_neighbor_lists(
            system, model.requested_neighbor_lists()
        )

        outputs = {"energy": ModelOutput(per_atom=False)}
        output = model([system], outputs)
        energy = output["energy"].block().values.sum()
        return energy

    assert torch.autograd.gradcheck(compute, positions, fast_mode=True)
    assert torch.autograd.gradgradcheck(compute, positions, fast_mode=True)

    if torch.cuda.is_available():
        positions_cuda = positions.detach().cuda().requires_grad_(True)
        assert torch.autograd.gradcheck(compute, positions_cuda, fast_mode=True)
        assert torch.autograd.gradgradcheck(compute, positions_cuda, fast_mode=True)


def test_autograd_cell():
    """Tests the basic functionality of the forward pass of the model."""
    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "energy": get_energy_target_info({"quantity": "energy", "unit": "eV"})
        },
    )

    model = NativePET(MODEL_HYPERS, dataset_info)
    model = model.to(torch.float64)

    cell = torch.eye(3, dtype=torch.float64, requires_grad=True)

    def compute(cell):
        system = System(
            types=torch.tensor([6, 6]),
            positions=torch.tensor(
                [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
                dtype=torch.float64,
                requires_grad=True,
            ),
            cell=cell,
            pbc=torch.tensor([True, True, True]),
        )

        system = get_system_with_neighbor_lists(
            system, model.requested_neighbor_lists()
        )

        outputs = {"energy": ModelOutput(per_atom=False)}
        output = model([system], outputs)
        energy = output["energy"].block().values.sum()
        return energy

    assert torch.autograd.gradcheck(compute, cell, fast_mode=True)
    assert torch.autograd.gradgradcheck(compute, cell, fast_mode=True)

    if torch.cuda.is_available():
        cell_cuda = cell.detach().cuda().requires_grad_(True)
        assert torch.autograd.gradcheck(compute, cell_cuda, fast_mode=True)
        assert torch.autograd.gradgradcheck(compute, cell_cuda, fast_mode=True)
