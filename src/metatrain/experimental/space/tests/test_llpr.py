import copy

import pytest

from metatrain.utils.testing import LLPRInterfaceTests
from metatrain.utils.testing.llpr import (
    cartesian_dataset_info,
    spherical_dataset_info,
    target_dataset_info,
)

from . import MODEL_HYPERS


def _cartesian_rank2_dataset_info():
    return target_dataset_info({"cartesian": {"rank": 2}}, "atom")


class TestLLPRInterface(LLPRInterfaceTests):
    architecture = "experimental.space"

    @pytest.fixture(
        params=[
            spherical_dataset_info,
            cartesian_dataset_info,
            _cartesian_rank2_dataset_info,
        ],
        ids=["spherical", "cartesian", "cartesian-rank2"],
    )
    def dataset_info(self, request):
        return request.param()

    @pytest.fixture
    def model_hypers(self) -> dict:
        hypers = copy.deepcopy(MODEL_HYPERS)
        hypers["num_element_channels"] = 2
        hypers["num_gnn_layers"] = 1
        hypers["num_tensor_products"] = 2
        # max_eigenvalue=25.0 gives l_max=2, needed for the lambda=2 block
        hypers["radial_basis"]["max_eigenvalue"] = 25.0
        hypers["radial_basis"]["mlp_expansion_ratio"] = 1
        hypers["radial_basis"]["mlp_depth"] = 2
        hypers["mlp_head_expansion_ratio"] = 1
        return hypers
