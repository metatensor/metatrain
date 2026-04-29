import copy
import warnings
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from metatrain.utils.architectures import check_architecture_options, get_default_hypers
from metatrain.utils.pydantic import MetatrainValidationError
from metatrain.utils.testing import ExportedTests, InputTests, TrainingTests


REPO_ROOT = Path(__file__).parents[5]


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
    hypers["tensor_basis_defaults"]["soap"]["max_radial"] = 1
    hypers["tensor_basis_defaults"]["soap"]["cutoff"]["radius"] = 3.0
    hypers["tensor_basis_defaults"]["soap"]["cutoff"]["width"] = 0.5
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


def _validate_architecture_options_file(path: Path) -> None:
    options = OmegaConf.load(path)
    architecture_name = options.architecture.name
    merged = OmegaConf.merge(
        {"architecture": get_default_hypers(architecture_name)}, options
    )
    architecture_options = OmegaConf.to_container(
        merged["architecture"], resolve=True
    )
    check_architecture_options(architecture_name, architecture_options)


def test_e_pet_tensor_basis_options_do_not_expose_max_angular() -> None:
    hypers = get_default_hypers("experimental.e_pet")

    assert "max_angular" not in hypers["model"]["tensor_basis_defaults"]["soap"]
    assert (
        "max_angular"
        not in hypers["model"]["tensor_basis_defaults"][
            "extra_l1_vector_basis_branches"
        ][0]
    )


def test_e_pet_option_files_validate_without_tensor_basis_max_angular() -> None:
    _validate_architecture_options_file(
        REPO_ROOT / "examples/1-advanced/options-e-pet-pet-omat-xs32.yaml"
    )
    _validate_architecture_options_file(
        REPO_ROOT / "src/metatrain/experimental/e_pet/tests/options-e-pet.yaml"
    )


def test_e_pet_training_defaults_use_split_learning_rates() -> None:
    training = get_default_hypers("experimental.e_pet")["training"]

    assert training["learning_rate"] == 2.0e-4
    assert training["pet_trunk_learning_rate"] == 2.0e-4
    assert training["tensor_basis_learning_rate"] == 1.0e-3
    assert training["readout_learning_rate"] == 1.0e-3


def test_e_pet_tensor_basis_rejects_max_angular() -> None:
    hypers = copy.deepcopy(get_default_hypers("experimental.e_pet"))
    hypers["model"]["tensor_basis_defaults"]["soap"]["max_angular"] = 2

    with pytest.raises(MetatrainValidationError, match="max_angular"):
        check_architecture_options("experimental.e_pet", hypers)


@pytest.mark.parametrize(
    "option_name",
    ("add_l1_species_dependent_vector", "l1_species_dependent_vector_soap"),
)
def test_e_pet_tensor_basis_rejects_removed_l1_species_options(
    option_name: str,
) -> None:
    hypers = copy.deepcopy(get_default_hypers("experimental.e_pet"))
    hypers["model"]["tensor_basis_defaults"][option_name] = {}

    with pytest.raises(MetatrainValidationError, match=option_name):
        check_architecture_options("experimental.e_pet", hypers)
