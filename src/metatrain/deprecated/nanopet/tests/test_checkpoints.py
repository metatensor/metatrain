import copy

import pytest

from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.testing import CheckpointTests


class TestCheckpoints(CheckpointTests):
    architecture = "deprecated.nanopet"

    @pytest.fixture
    def minimal_model_hypers(self, model_hypers):
        hypers = copy.deepcopy(model_hypers)
        hypers["d_pet"] = 1
        hypers["num_heads"] = 1
        hypers["num_attention_layers"] = 1
        hypers["num_gnn_layers"] = 1
        return hypers

    @pytest.mark.parametrize("context", ["finetune", "restart", "export"])
    def test_get_checkpoint(self, context, model_hypers):
        """
        Test that the checkpoint created by the model.get_checkpoint()
        function can be loaded back in all possible contexts.
        """
        dataset_info = DatasetInfo(
            length_unit="Angstrom",
            atomic_types=[1, 6, 7, 8],
            targets={"energy": get_energy_target_info("energy", {"unit": "eV"})},
        )
        model = self.model_cls(model_hypers, dataset_info)
        checkpoint = model.get_checkpoint()
        self.model_cls.load_checkpoint(checkpoint, context)
