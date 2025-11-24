from pathlib import Path
from typing import Any

import pytest
import torch

from metatrain.utils.abc import TrainerInterface
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
    """This is the base class for all architecture tests.

    It doesn't implement any tests itself, but provides fixtures
    and helper functions that are generally useful for testing
    architectures.

    Child classes can override everything, including fixtures, to
    make the tests suit their needs. Note that some fixtures defined
    here depend on other fixtures, but when overriding them, you can
    change completely their signature.
    """

    architecture: str
    """Name of the architecture to be tested.

    Based on this, the test suite will find the model and trainer classes
    as well as the hyperparameters.
    """

    @pytest.fixture
    def dataset_path(self) -> str:
        """Fixture that provides a path to a dataset file for testing.

        :return: The path to the dataset file.
        """
        return str(Path(__file__).parents[4] / "tests/resources/qm9_reduced_100.xyz")

    @pytest.fixture
    def dataset_targets(self, dataset_path: str) -> dict[str, dict]:
        """Fixture that provides the target hyperparameters for the dataset used
        in testing.

        :param dataset_path: The path to the dataset file.
        :return: A dictionary with target hyperparameters.
        """
        energy_target = {
            "quantity": "energy",
            "read_from": dataset_path,
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
        self, dataset_targets: dict[str, dict], dataset_path: str
    ) -> tuple[Dataset, dict[str, TargetInfo], DatasetInfo]:
        """Helper function to load the dataset used in testing.

        :param dataset_targets: The target hyperparameters for the dataset.
        :param dataset_path: The path to the dataset file.
        :return: A tuple containing the dataset, target info, and dataset info.
        """
        dataset, targets_info, _ = get_dataset(
            {
                "systems": {
                    "read_from": dataset_path,
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
        """Fixture to provide the torch device for testing.

        :param request: The pytest request fixture.
        :return: The torch device to be used.
        """
        device = request.param
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA is not available")
        return torch.device(device)

    @pytest.fixture(params=[torch.float32, torch.float64])
    def dtype(self, request: pytest.FixtureRequest) -> torch.dtype:
        """Fixture to provide the model data type for testing.

        :param request: The pytest request fixture.
        :return: The torch data type to be used.
        """
        return request.param

    @pytest.fixture
    def dataset_info(self) -> DatasetInfo:
        """Fixture that provides a basic ``DatasetInfo`` with an
        energy target for testing.

        :return: A ``DatasetInfo`` instance with an energy target.
        """
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
    def per_atom(self, request: pytest.FixtureRequest) -> bool:
        """Fixture to test both per-atom and per-system targets.

        :param request: The pytest request fixture.
        :return: Whether the target is per-atom or not.
        """
        return request.param

    @pytest.fixture
    def dataset_info_scalar(self, per_atom: bool) -> DatasetInfo:
        """Fixture that provides a basic ``DatasetInfo`` with a scalar target
        for testing.

        :param per_atom: Whether the target is per-atom or not.
        :return: A ``DatasetInfo`` instance with a scalar target.
        """
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
    def dataset_info_vector(self, per_atom: bool) -> DatasetInfo:
        """Fixture that provides a basic ``DatasetInfo`` with a vector target
        for testing.

        :param per_atom: Whether the target is per-atom or not.
        :return: A ``DatasetInfo`` instance with a vector target.
        """
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
    def o3_lambda(self, request: pytest.FixtureRequest) -> int:
        """Fixture to provide different O(3) lambda values for
        testing spherical tensors.

        :param request: The pytest request fixture.
        :return: The O(3) lambda value.
        """
        return request.param

    @pytest.fixture(params=[-1, 1])
    def o3_sigma(self, request: pytest.FixtureRequest) -> int:
        """Fixture to provide different O(3) sigma values for
        testing spherical tensors.

        :param request: The pytest request fixture.
        :return: The O(3) sigma value.
        """
        return request.param

    @pytest.fixture
    def dataset_info_spherical(self, o3_lambda: int, o3_sigma: int) -> DatasetInfo:
        """Fixture that provides a basic ``DatasetInfo`` with a
        spherical target for testing.

        :param o3_lambda: The O(3) lambda of the spherical target.
        :param o3_sigma: The O(3) sigma of the spherical target.
        :return: A ``DatasetInfo`` instance with a spherical target.
        """
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
    def dataset_info_multispherical(self, per_atom: bool) -> DatasetInfo:
        """Fixture that provides a basic ``DatasetInfo`` with multiple spherical
        targets for testing.

        :param per_atom: Whether the target is per-atom or not.
        :return: A ``DatasetInfo`` instance with a multiple spherical targets.
        """
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

    # Replace the Any type hint with type[ModelInterface]
    # once https://github.com/metatensor/metatrain/issues/942 is solved.
    @property
    def model_cls(self) -> Any:
        """The model class to be tested."""
        architecture = import_architecture(self.architecture)
        return architecture.__model__

    @property
    def trainer_cls(self) -> type[TrainerInterface]:
        """The trainer class to be tested."""
        architecture = import_architecture(self.architecture)
        return architecture.__trainer__

    @pytest.fixture
    def default_hypers(self) -> dict:
        """Fixture that provides the default hyperparameters for testing.

        :return: The default hyperparameters for the architecture.
        """
        return get_default_hypers(self.architecture)

    @pytest.fixture
    def model_hypers(self) -> dict:
        """Fixture that provides the model hyperparameters for testing.

        If not overriden, these are the default model hyperparameters.

        :return: The model hyperparameters for testing.
        """
        return get_default_hypers(self.architecture)["model"]

    @pytest.fixture
    def minimal_model_hypers(self) -> dict:
        """The hypers that produce the smallest possible model.

        This should be overridden in each architecture test class
        to ensure that the tests run quickly/checkpoints occupy
        little disk space.

        :return: The minimal model hyperparameters for testing.
        """
        return get_default_hypers(self.architecture)["model"]
