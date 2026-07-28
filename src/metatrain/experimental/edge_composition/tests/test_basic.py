# mypy: disable-error-code="override"
import pytest

from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_generic_target_info
from metatrain.utils.testing import (
    ArchitectureTests,
    TorchscriptTests,
)


class EdgeCompositionTests(ArchitectureTests):
    architecture = "experimental.edge_composition"

    @pytest.fixture
    def dataset_info(self) -> DatasetInfo:
        """Fixture that provides a basic ``DatasetInfo`` with a spherical
        target that uses an atomic basis for testing.

        :return: A ``DatasetInfo`` instance with a spherical target in
            an atomic basis.
        """
        return DatasetInfo(
            length_unit="Angstrom",
            atomic_types=[1, 6, 7, 8],
            targets={
                "spherical_tensor_atomic_basis": get_generic_target_info(
                    "spherical_tensor_atomic_basis",
                    {
                        "quantity": "spherical_tensor_atomic_basis",
                        "unit": "",
                        "type": {
                            "spherical": {
                                "product": "cartesian",
                                "irreps": {
                                    1: [
                                        {"num": 2, "o3_lambda": 0, "o3_sigma": 1},
                                        {"num": 1, "o3_lambda": 1, "o3_sigma": 1},
                                    ],
                                    6: [
                                        {"num": 3, "o3_lambda": 0, "o3_sigma": 1},
                                        {"num": 2, "o3_lambda": 1, "o3_sigma": 1},
                                        {"num": 1, "o3_lambda": 2, "o3_sigma": 1},
                                    ],
                                    7: [
                                        {"num": 3, "o3_lambda": 0, "o3_sigma": 1},
                                        {"num": 2, "o3_lambda": 1, "o3_sigma": 1},
                                        {"num": 1, "o3_lambda": 2, "o3_sigma": 1},
                                    ],
                                    8: [
                                        {"num": 3, "o3_lambda": 0, "o3_sigma": 1},
                                        {"num": 2, "o3_lambda": 1, "o3_sigma": 1},
                                        {"num": 1, "o3_lambda": 2, "o3_sigma": 1},
                                    ],
                                },
                            }
                        },
                        "num_subtargets": 1,
                        "sample_kind": "atom_pair",
                    },
                )
            },
        )


class TestTorchscript(TorchscriptTests, EdgeCompositionTests):
    pass
