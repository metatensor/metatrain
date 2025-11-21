from pathlib import Path

import pytest
import torch

from metatrain.utils.architectures import get_default_hypers, import_architecture
from metatrain.utils.data import (
    Dataset,
    DatasetInfo,
    TargetInfo,
    get_atomic_types,
    get_dataset,
)
from metatrain.utils.data.target_info import (
    get_energy_target_info,
    get_generic_target_info,
)


class ArchitectureTests:
    architecture: str

    @pytest.fixture
    def DATASET_PATH(self) -> str:
        """Path to a dataset file for testing.

        :return: The path to the dataset file.
        """
        return str(Path(__file__).parents[4] / "tests/resources/qm9_reduced_100.xyz")

    @pytest.fixture
    def dataset_targets(self, DATASET_PATH: str) -> dict[str, dict]:
        """Target hyperparameters for the dataset used in testing.

        :param DATASET_PATH: The path to the dataset file.
        :return: A dictionary with target hyperparameters.
        """
        energy_target = {
            "quantity": "energy",
            "read_from": DATASET_PATH,
            "reader": "ase",
            "key": "U0",
            "unit": "eV",
            "type": "scalar",
            "per_atom": False,
            "num_subtargets": 1,
            "forces": False,
            "stress": False,
            "virial": False,
        }

        return {"energy": energy_target}

    def get_dataset(
        self, dataset_targets: dict[str, dict], DATASET_PATH: str
    ) -> tuple[Dataset, dict[str, TargetInfo], DatasetInfo]:
        """Helper function to load the dataset used in testing.

        :param dataset_targets: The target hyperparameters for the dataset.
        :param DATASET_PATH: The path to the dataset file.
        :return: A tuple containing the dataset, target info, and dataset info.
        """
        dataset, targets_info, _ = get_dataset(
            {
                "systems": {
                    "read_from": DATASET_PATH,
                    "reader": "ase",
                },
                "targets": dataset_targets,
            }
        )

        dataset_info = DatasetInfo(
            length_unit="",
            atomic_types=get_atomic_types(dataset),
            targets=targets_info,
        )

        return dataset, targets_info, dataset_info

    @pytest.fixture(params=("cpu", "cuda"))
    def device(self, request: pytest.FixtureRequest) -> torch.device:
        """Fixture to provide the device for testing."""
        device = request.param
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA is not available")
        return torch.device(device)

    @pytest.fixture(params=[torch.float32, torch.float64])
    def dtype(self, request: pytest.FixtureRequest) -> torch.dtype:
        """Fixture to provide the model data type for testing."""
        return request.param

    @pytest.fixture
    def dataset_info(self) -> DatasetInfo:
        """Provides a basic DatasetInfo with an energy for testing."""
        return DatasetInfo(
            length_unit="Angstrom",
            atomic_types=[1, 6, 7, 8],
            targets={
                "energy": get_energy_target_info(
                    "energy", {"quantity": "energy", "unit": "eV"}
                )
            },
        )

    @pytest.fixture(params=[True, False])
    def per_atom(self, request):
        return request.param

    @pytest.fixture
    def dataset_info_scalar(self, per_atom):
        return DatasetInfo(
            length_unit="Angstrom",
            atomic_types=[1, 6, 7, 8],
            targets={
                "scalar": get_generic_target_info(
                    "scalar",
                    {
                        "quantity": "scalar",
                        "unit": "",
                        "type": "scalar",
                        "num_subtargets": 5,
                        "per_atom": per_atom,
                    },
                )
            },
        )

    @pytest.fixture
    def dataset_info_vector(self, per_atom):
        return DatasetInfo(
            length_unit="Angstrom",
            atomic_types=[1, 6, 7, 8],
            targets={
                "vector": get_generic_target_info(
                    "vector",
                    {
                        "quantity": "vector",
                        "unit": "",
                        "type": {"cartesian": {"rank": 1}},
                        "num_subtargets": 5,
                        "per_atom": per_atom,
                    },
                )
            },
        )

    @pytest.fixture(params=[0, 1, 2, 3])
    def o3_lambda(self, request):
        return request.param

    @pytest.fixture(params=[-1, 1])
    def o3_sigma(self, request):
        return request.param

    @pytest.fixture
    def dataset_info_spherical(self, o3_lambda, o3_sigma):
        """Tests that the spherical modules can be jitted."""

        return DatasetInfo(
            length_unit="Angstrom",
            atomic_types=[1, 6, 7, 8],
            targets={
                "spherical_target": get_generic_target_info(
                    "spherical_target",
                    {
                        "quantity": "",
                        "unit": "",
                        "type": {
                            "spherical": {
                                "irreps": [
                                    {"o3_lambda": o3_lambda, "o3_sigma": o3_sigma}
                                ]
                            }
                        },
                        "num_subtargets": 5,
                        "per_atom": False,
                    },
                )
            },
        )

    @pytest.fixture
    def dataset_info_multispherical(self, per_atom):
        return DatasetInfo(
            length_unit="Angstrom",
            atomic_types=[1, 6, 7, 8],
            targets={
                "spherical_tensor": get_generic_target_info(
                    "spherical_tensor",
                    {
                        "quantity": "spherical_tensor",
                        "unit": "",
                        "type": {
                            "spherical": {
                                "irreps": [
                                    {"o3_lambda": 2, "o3_sigma": 1},
                                    {"o3_lambda": 1, "o3_sigma": 1},
                                    {"o3_lambda": 0, "o3_sigma": 1},
                                ]
                            }
                        },
                        "num_subtargets": 100,
                        "per_atom": per_atom,
                    },
                )
            },
        )

    @property
    def model_cls(self):
        architecture = import_architecture(self.architecture)
        return architecture.__model__

    @property
    def trainer_cls(self):
        architecture = import_architecture(self.architecture)
        return architecture.__trainer__

    @pytest.fixture
    def default_hypers(self):
        return get_default_hypers(self.architecture)

    @pytest.fixture
    def model_hypers(self):
        return get_default_hypers(self.architecture)["model"]

    @pytest.fixture
    def minimal_model_hypers(self):
        """The hypers that produce the smallest possible model.

        This should be overridden in each architecture test class
        to ensure that the tests run quickly/checkpoints occupy
        little disk space.
        """
        return get_default_hypers(self.architecture)["model"]
