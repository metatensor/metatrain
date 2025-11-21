import torch
from metatomic.torch import ModelEvaluationOptions, ModelMetadata, System

from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)

from .base import ArchitectureTests


class ExportedTests(ArchitectureTests):
    def test_to(self, device, dtype, model_hypers, dataset_info):
        """Tests that the `.to()` method of the exported model works."""

        model = self.model_cls(model_hypers, dataset_info).to(dtype=dtype)

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
