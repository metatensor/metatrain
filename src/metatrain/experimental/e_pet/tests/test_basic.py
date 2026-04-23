import copy
import warnings

import pytest
import torch

from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.testing import ExportedTests, InputTests, TrainingTests


def _minimal_model_hypers() -> dict:
    hypers = copy.deepcopy(get_default_hypers("experimental.e_pet")["model"])
    hypers["pet"]["cutoff"] = 3.0
    hypers["pet"]["cutoff_width"] = 0.5
    hypers["pet"]["d_pet"] = 4
    hypers["pet"]["d_head"] = 4
    hypers["pet"]["d_node"] = 4
    hypers["pet"]["d_feedforward"] = 8
    hypers["pet"]["num_heads"] = 1
    hypers["pet"]["num_attention_layers"] = 1
    hypers["pet"]["num_gnn_layers"] = 1
    hypers["pet"]["activation"] = "SiLU"
    hypers["pet"]["featurizer_type"] = "residual"
    hypers["tensor_basis_defaults"]["soap"]["max_angular"] = 2
    hypers["tensor_basis_defaults"]["soap"]["max_radial"] = 1
    hypers["tensor_basis_defaults"]["soap"]["cutoff"]["radius"] = 3.0
    hypers["tensor_basis_defaults"]["soap"]["cutoff"]["width"] = 0.5
    hypers["tensor_basis_defaults"]["l1_species_dependent_vector_soap"] = copy.deepcopy(
        hypers["tensor_basis_defaults"]["soap"]
    )
    hypers["tensor_basis_defaults"]["extra_l1_vector_basis_branches"] = [
        copy.deepcopy(hypers["tensor_basis_defaults"]["soap"])
    ]
    return hypers


@pytest.mark.filterwarnings("ignore:custom data:UserWarning")
class EPETArchitectureTests:
    architecture = "experimental.e_pet"

    @pytest.fixture(params=("cpu", "cuda"))
    def device(self, request: pytest.FixtureRequest) -> torch.device:
        device = request.param
        if device == "cuda":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    cuda_available = torch.cuda.is_available()
                except Exception:
                    cuda_available = False
            if not cuda_available:
                pytest.skip("CUDA is not available")
        return torch.device(device)

    @pytest.fixture
    def model_hypers(self):
        return _minimal_model_hypers()

    @pytest.fixture
    def minimal_model_hypers(self):
        return _minimal_model_hypers()

    @pytest.fixture
    def default_hypers(self):
        hypers = copy.deepcopy(get_default_hypers(self.architecture))
        hypers["training"]["batch_size"] = 2
        hypers["training"]["num_epochs"] = 1
        hypers["training"]["checkpoint_interval"] = 1
        hypers["training"]["log_interval"] = 1
        return hypers


class TestInput(EPETArchitectureTests, InputTests):
    pass


class TestExported(EPETArchitectureTests, ExportedTests):
    pass


class TestTraining(EPETArchitectureTests, TrainingTests):
    pass
