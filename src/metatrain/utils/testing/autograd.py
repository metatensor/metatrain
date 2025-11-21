import torch
from metatomic.torch import ModelOutput, System

from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from .base import ArchitectureTests


class AutogradTests(ArchitectureTests):
    """Tests that autograd works correctly for a given model."""

    def test_autograd_positions(self, device, model_hypers, dataset_info):
        """Tests the basic functionality of the forward pass of the model."""

        device = torch.device(device)

        model = self.model_cls(model_hypers, dataset_info)
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

    def test_autograd_cell(self, device, model_hypers, dataset_info):
        """Tests the basic functionality of the forward pass of the model."""

        device = torch.device(device)

        model = self.model_cls(model_hypers, dataset_info)
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
