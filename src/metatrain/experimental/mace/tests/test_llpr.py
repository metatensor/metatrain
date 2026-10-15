import copy
import re

import pytest

from metatrain.utils.testing import LLPRInterfaceTests
from metatrain.utils.testing.llpr import (
    setup_backbone,
    spherical_dataset_info,
    wrap_backbone,
)

from . import MODEL_HYPERS


class TestLLPRInterface(LLPRInterfaceTests):
    architecture = "experimental.mace"

    @pytest.fixture
    def model_hypers(self) -> dict:
        hypers = copy.deepcopy(MODEL_HYPERS)
        hypers["hidden_irreps"] = "4x0e + 4x1o + 4x2e"
        hypers["MLP_irreps"] = "4x0e"
        hypers["num_interactions"] = 2
        return hypers

    def test_target_with_a_missing_irrep_is_refused(self, model_hypers):
        """Check a target whose irrep MACE does not have is refused."""
        # 1e is masked to zero in MACE's hidden features
        dataset_info = spherical_dataset_info(irreps=[{"o3_lambda": 1, "o3_sigma": -1}])
        backbone = setup_backbone(self.architecture, model_hypers, dataset_info)
        message = (
            "The wrapped model exposes no last-layer features, so LLPR can not compute "
            "any uncertainty"
        )
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            wrap_backbone(backbone, dataset_info, ensembles=False)
