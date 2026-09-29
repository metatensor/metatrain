import copy

import torch
from metatomic.torch import ModelOutput

from metatrain.experimental.space import SPACE
from metatrain.utils.testing import LLPRInterfaceTests
from metatrain.utils.testing.llpr import (
    UNCERTAINTY,
    fit_llpr,
    make_systems,
    rotate_system,
    target_dataset_info,
    wrap_backbone,
)

from . import MODEL_HYPERS


def _make_hypers() -> dict:
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


def _backbone(dataset_info):
    return SPACE(_make_hypers(), dataset_info).to(torch.float64)


class TestLLPRInterface(LLPRInterfaceTests):
    def make_backbone(self, dataset_info):
        return _backbone(dataset_info)


def test_cartesian_rank2_uncertainty_via_shared_features():
    """Check a rank-2 Cartesian target, which reads the shared invariant features,
    gets the same rotation-invariant uncertainty for all its components."""
    dataset_info = target_dataset_info({"cartesian": {"rank": 2}}, "atom")
    model = wrap_backbone(_backbone(dataset_info), dataset_info, ensembles=False)
    systems = make_systems(model.model, 4)
    fit_llpr(model, systems)
    outputs = {UNCERTAINTY: ModelOutput(sample_kind="atom")}
    unc = model([systems[0]], outputs)[UNCERTAINTY].block().values
    rotated_system = rotate_system(model.model, systems[0])
    unc_rot = model([rotated_system], outputs)[UNCERTAINTY].block().values
    torch.testing.assert_close(unc, unc[:, :1, :1].expand_as(unc))
    torch.testing.assert_close(unc, unc_rot)


def test_torchscript_with_no_block_aligned_targets():
    """Check a model with an empty `last_layer_feature_sizes` still scripts."""
    dataset_info = target_dataset_info({"cartesian": {"rank": 2}}, "atom")
    backbone = SPACE(_make_hypers(), dataset_info)
    assert backbone.last_layer_feature_sizes == {}
    backbone.prepare_for_export()
    torch.jit.script(backbone)
