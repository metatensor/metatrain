import copy
import re

import pytest
import torch

from metatrain.experimental.mace import MetaMACE
from metatrain.utils.testing import LLPRInterfaceTests
from metatrain.utils.testing.llpr import TARGET, spherical_dataset_info, wrap_backbone

from . import MODEL_HYPERS


def _make_hypers() -> dict:
    hypers = copy.deepcopy(MODEL_HYPERS)
    hypers["hidden_irreps"] = "4x0e + 4x1o + 4x2e"
    hypers["MLP_irreps"] = "4x0e"
    hypers["num_interactions"] = 2
    return hypers


def _backbone(dataset_info):
    return MetaMACE(_make_hypers(), dataset_info).to(torch.float64)


class TestLLPRInterface(LLPRInterfaceTests):
    def make_backbone(self, dataset_info):
        return _backbone(dataset_info)


def test_missing_irrep_block_refuses_wrapping_for_ensembles():
    """Check a target whose irrep is absent from MACE's hidden features (here
    1e, masked to zero) exposes no block-aligned features: ensembles are refused,
    and without ensembles the target simply has no uncertainty output."""
    dataset_info = spherical_dataset_info(irreps=[{"o3_lambda": 1, "o3_sigma": -1}])
    message = f"Output '{TARGET}' in ensembles section is not supported by the model"
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        wrap_backbone(_backbone(dataset_info), dataset_info, ensembles=True)
    model = wrap_backbone(_backbone(dataset_info), dataset_info, ensembles=False)
    assert TARGET not in model.outputs_list
