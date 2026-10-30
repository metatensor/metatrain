"""
Tests for the ``d_head``, ``head_type``, ``num_head_layers`` and
``readout_type`` hyperparameters of PET.

The target is an atomic-basis target, densified as during training: one block per
``o3_lambda``, whose properties are padded to the largest basis among atomic types.
"""

import copy
import gzip

import pytest
import torch
from metatomic.torch import ModelOutput, System
from omegaconf import OmegaConf

from metatrain.pet import PET
from metatrain.utils.architectures import (
    check_architecture_options,
    get_default_hypers,
)
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.atomic_basis_helpers import (
    densify_atomic_basis_dataset_info,
)
from metatrain.utils.data.target_info import get_generic_target_info
from metatrain.utils.io import model_from_checkpoint
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from . import MODEL_HYPERS


TARGET = "mtt::basis"
# H has an l=0 basis, O an l=0 and an l=1 basis: two blocks once densified
IRREPS = {
    1: [{"o3_lambda": 0, "o3_sigma": 1, "num": 2}],
    8: [
        {"o3_lambda": 0, "o3_sigma": 1, "num": 4},
        {"o3_lambda": 1, "o3_sigma": 1, "num": 3},
    ],
}
NUM_BLOCKS = 2


def _dataset_info():
    target_info = get_generic_target_info(
        TARGET,
        {
            "quantity": "",
            "unit": "",
            "type": {"spherical": {"irreps": IRREPS}},
            "num_subtargets": 1,
            "sample_kind": "atom",
        },
    )
    dataset_info = DatasetInfo(
        length_unit="Angstrom", atomic_types=[1, 8], targets={TARGET: target_info}
    )
    return densify_atomic_basis_dataset_info(dataset_info)


def _model(**hypers):
    model_hypers = copy.deepcopy(MODEL_HYPERS)
    model_hypers.update(hypers)
    return PET(model_hypers, _dataset_info()).eval()


def _system(model):
    """A water molecule (atoms O, H, H)."""
    system = System(
        types=torch.tensor([8, 1, 1]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.9, 0.0], [0.9, 0.0, 0.0]]),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )
    return get_system_with_neighbor_lists(system, model.requested_neighbor_lists())


def _predict(model, system):
    """The predictions for ``system``, one tensor per block."""
    output = model([system], {TARGET: ModelOutput(sample_kind="atom")})[TARGET]
    return [block.values.detach() for block in output.blocks()]


def test_torchscript():
    """Check the scripted model, with per-block heads and conditioned readouts."""
    model = _model(head_type="per_block", readout_type="one-hot")
    system = _system(model)
    for scripted, eager in zip(
        _predict(torch.jit.script(model), system), _predict(model, system), strict=True
    ):
        torch.testing.assert_close(scripted, eager)


@pytest.mark.parametrize("head_type", ["per_target", "per_block"])
@pytest.mark.parametrize("readout_type", [None, "one-hot"])
def test_predictions_are_linear_in_last_layer_features(head_type, readout_type):
    """Check the predictions without biases against the last-layer features."""
    model = _model(head_type=head_type, readout_type=readout_type)
    with torch.no_grad():
        for name, parameter in model.backend.named_parameters():
            if "last_layers" in name and name.endswith("bias"):
                parameter.zero_()

    system = _system(model)
    llf_name = "mtt::aux::basis_last_layer_features"
    output = model(
        [system],
        {
            TARGET: ModelOutput(sample_kind="atom"),
            llf_name: ModelOutput(sample_kind="atom"),
        },
    )
    features = output[llf_name].block()

    for index, (key, block) in enumerate(output[TARGET].items()):
        # the node and edge weights of each readout layer, in the order of the features
        weights = []
        for layer in range(model.num_readout_layers):
            for which in ["node", "edge"]:
                last_layers = getattr(model.backend, f"{which}_last_layers")
                weights.append(list(last_layers[TARGET][layer].values())[index].weight)
        weight = torch.cat(weights, dim=-1)
        if readout_type is None:
            weight = weight.unsqueeze(0)
        # the weights of the type of each atom (O, H, H)
        weight = weight[[1, 0, 0] if readout_type == "one-hot" else [0, 0, 0]]

        # with per-block heads, the features labelled by the key of the block
        values = features.values
        if head_type == "per_block":
            mask = torch.ones(len(features.properties), dtype=torch.bool)
            for name in key.names:
                mask &= features.properties.column(name) == key[name]
            values = values[:, mask]
        torch.testing.assert_close(
            block.values.reshape(len(block.samples), -1),
            torch.einsum("sof,sf->so", weight, values),
        )


def test_per_block_heads_are_independent():
    """Check changing the head of the last block only changes its predictions."""
    model = _model(head_type="per_block")
    system = _system(model)
    before = _predict(model, system)
    with torch.no_grad():
        for parameter in model.backend.node_heads[TARGET][NUM_BLOCKS - 1].parameters():
            parameter += 1.0
    after = _predict(model, system)

    torch.testing.assert_close(after[0], before[0])
    assert not torch.allclose(after[1], before[1])


def test_d_head_and_num_head_layers():
    model = _model(d_head={"node": 8, "edge": 16}, num_head_layers=3)
    assert len(model.backend.node_heads[TARGET][0]) == 2 * 3  # Linear + SiLU
    ll_features = model.outputs["mtt::aux::basis_last_layer_features"].layout
    assert len(ll_features.block().properties) == model.num_readout_layers * (8 + 16)


def test_invalid_num_head_layers():
    with pytest.raises(ValueError, match="^num_head_layers must be >= 1, got 0$"):
        _model(num_head_layers=0)


def test_per_target_readout_type():
    model = _model(readout_type={TARGET: "one-hot"})
    assert model.last_layer_depends_on_atomic_type(TARGET)
    # targets that are not listed use the default
    model = _model(readout_type={"mtt::other": "one-hot"})
    assert not model.last_layer_depends_on_atomic_type(TARGET)


def _check_options(hypers):
    """Merge options set by users with the defaults, as ``mtt train`` does, and
    validate them."""
    options = OmegaConf.merge(
        {"name": "pet", **get_default_hypers("pet")}, {"model": hypers}
    )
    check_architecture_options("pet", OmegaConf.to_container(options))


def test_per_target_readout_type_option_is_valid():
    _check_options({"readout_type": {TARGET: "one-hot"}})


def test_v18_checkpoint_upgrade():
    """Check a checkpoint from before these hypers loads with all of its weights."""
    with gzip.open("checkpoints/model-v18_trainer-v15.ckpt.gz", "rb") as fd:
        checkpoint = torch.load(fd, weights_only=False)
    state_dict = checkpoint["core"]["best_model_state_dict"]

    model = model_from_checkpoint(checkpoint, "export").core
    loaded = model.state_dict()
    assert state_dict.keys() == loaded.keys()
    for name, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(loaded[name], value, rtol=0, atol=0)
