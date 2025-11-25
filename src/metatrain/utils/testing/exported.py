import torch
from metatomic.torch import ModelEvaluationOptions, ModelMetadata, System

from metatrain.utils.data import DatasetInfo
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)

from .architectures import ArchitectureTests


class ExportedTests(ArchitectureTests):
    """Test suite to test exported models."""

    def test_to(
        self,
        device: torch.device,
        dtype: torch.dtype,
        model_hypers: dict,
        dataset_info: DatasetInfo,
    ) -> None:
        """Tests that the `.to()` method of the exported model works.

        In other words, it tests that the exported model can be moved to
        different devices and dtypes.

        :param device: The device to move the exported model to.
        :param dtype: The dtype to move the exported model to.
        :param model_hypers: Hyperparameters to initialize the model.
        :param dataset_info: Dataset information to initialize the model.
        """

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
