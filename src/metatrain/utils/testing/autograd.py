import torch
from metatomic.torch import ModelOutput, System

from metatrain.utils.data import DatasetInfo
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from .architectures import ArchitectureTests


class AutogradTests(ArchitectureTests):
    """Tests that autograd works correctly for a given model."""

    cuda_nondet_tolerance = 0.0
    """Some operations in your model might be nondeterministic in CuBLAS.

    This can result in small differences in two gradient computations
    with the same input and outputs. This number sets the nondeterministic
    tolerance for ``gradcheck`` and ``gradgradcheck`` when running on CUDA.
    """

    def test_autograd_positions(
        self, device: torch.device, model_hypers: dict, dataset_info: DatasetInfo
    ) -> None:
        """Tests that autograd can compute gradients with respect to
        positions.

        It checks both first and second derivatives.

        It uses ``torch.autograd.gradcheck`` and
        ``torch.autograd.gradgradcheck`` for this purpose.

        :param device: The device to run the test on.
        :param model_hypers: Hyperparameters to initialize the model.
        :param dataset_info: Dataset information to initialize the model.
        """

        device = torch.device(device)

        nondet_tolerance = self.cuda_nondet_tolerance if device.type == "cuda" else 0.0

        model = self.model_cls(model_hypers, dataset_info)
        model = model.to(dtype=torch.float64, device=device)

        def compute(positions: torch.Tensor) -> torch.Tensor:
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
        assert torch.autograd.gradcheck(
            compute, positions, fast_mode=True, nondet_tol=nondet_tolerance
        )
        assert torch.autograd.gradgradcheck(
            compute, positions, fast_mode=True, nondet_tol=nondet_tolerance
        )

    def test_autograd_cell(
        self, device: torch.device, model_hypers: dict, dataset_info: DatasetInfo
    ) -> None:
        """Tests that autograd can compute gradients with respect to
        the cell.

        It checks both first and second derivatives.

        It uses ``torch.autograd.gradcheck`` and
        ``torch.autograd.gradgradcheck`` for this purpose.

        :param device: The device to run the test on.
        :param model_hypers: Hyperparameters to initialize the model.
        :param dataset_info: Dataset information to initialize the model.
        """

        device = torch.device(device)
        nondet_tolerance = self.cuda_nondet_tolerance if device.type == "cuda" else 0.0

        model = self.model_cls(model_hypers, dataset_info)
        model = model.to(dtype=torch.float64, device=device)

        def compute(cell: torch.Tensor) -> torch.Tensor:
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

        assert torch.autograd.gradcheck(
            compute, cell, fast_mode=True, nondet_tol=nondet_tolerance
        )
        assert torch.autograd.gradgradcheck(
            compute, cell, fast_mode=True, nondet_tol=nondet_tolerance
        )
