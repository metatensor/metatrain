import pytest
import torch
from pathlib import Path

from metatrain.utils.architectures import import_architecture, get_default_hypers
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import (
    get_energy_target_info,
    get_generic_target_info
)

class ArchitectureTests:

    architecture: str

    @pytest.fixture
    def DATASET_PATH(self):
        return str(Path(__file__).parents[4] / "tests/resources/qm9_reduced_100.xyz")

    @pytest.fixture(params=("cpu", "cuda"))
    def device(self, request):
        device = request.param
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA is not available")
        return torch.device(device)
    
    @pytest.fixture(params=[torch.float32, torch.float64])
    def dtype(self, request):
        return request.param
    
    @pytest.fixture
    def dataset_info(self):
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
                                "irreps": [{"o3_lambda": o3_lambda, "o3_sigma": o3_sigma}]
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

    @pytest.fixture
    def model_hypers(self):
        return get_default_hypers(self.architecture)["model"]