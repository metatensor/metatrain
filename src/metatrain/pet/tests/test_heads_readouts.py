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
from metatrain.utils.pydantic import MetatrainValidationError

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


def _predict(model):
    """Predictions for a water molecule (atoms O, H, H), one tensor per block."""
    system = System(
        types=torch.tensor([8, 1, 1]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.9, 0.0], [0.9, 0.0, 0.0]]),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )
    system = get_system_with_neighbor_lists(system, model.requested_neighbor_lists())
    output = model([system], {TARGET: ModelOutput(sample_kind="atom")})[TARGET]
    return [block.values.detach() for block in output.blocks()]


def _last_layer(model, which="node"):
    """The last layer of the first block and readout layer of the target."""
    last_layers = getattr(model.backend, f"{which}_last_layers")[TARGET][0]
    return next(iter(last_layers.values()))


@pytest.mark.parametrize("head_type", ["per_target", "per_block"])
@pytest.mark.parametrize("gating", [False, "one-hot"])
def test_predictions_and_torchscript(head_type, gating):
    model = _model(head_type=head_type, readout_type={"atom_type_gating": gating})
    predictions = _predict(model)
    assert [p.shape for p in predictions] == [(3, 1, 4), (3, 3, 3)]
    assert all(torch.isfinite(p).all() for p in predictions)
    torch.jit.script(model)


def test_per_block_heads_are_independent():
    """Changing the head of the last block only changes the predictions of it."""
    model = _model(head_type="per_block")
    before = _predict(model)
    with torch.no_grad():
        for parameter in model.backend.node_heads[TARGET][NUM_BLOCKS - 1].parameters():
            parameter += 1.0
    after = _predict(model)

    torch.testing.assert_close(after[0], before[0])
    assert not torch.allclose(after[1], before[1])


@pytest.mark.parametrize("which", ["node", "edge"])
def test_one_hot_gating_uses_the_central_atom_type(which):
    """Changing the oxygen weights of a last layer changes the predictions of the
    oxygen atom only."""
    model = _model(readout_type={"atom_type_gating": "one-hot"})
    last_layer = _last_layer(model, which)
    assert last_layer.weight.shape[0] == 2  # one weight per atomic type: H and O

    before = _predict(model)[0]
    with torch.no_grad():
        last_layer.weight[1] += 1.0
    after = _predict(model)[0]

    assert not torch.allclose(after[0], before[0])  # O
    torch.testing.assert_close(after[1:], before[1:])  # H, H


def test_d_head_and_num_head_layers():
    model = _model(d_head={"node": 8, "edge": 16}, num_head_layers=3)
    node_head = model.backend.node_heads[TARGET][0]
    edge_head = model.backend.edge_heads[TARGET][0]
    assert len(node_head) == len(edge_head) == 6  # Linear+SiLU per layer
    assert node_head[-2].out_features == _last_layer(model, "node").in_features == 8
    assert edge_head[-2].out_features == _last_layer(model, "edge").in_features == 16
    assert model.last_layer_feature_size == model.num_readout_layers * (8 + 16)
    _predict(model)


def test_invalid_num_head_layers():
    with pytest.raises(ValueError, match="^num_head_layers must be >= 1, got 0$"):
        _model(num_head_layers=0)


def test_per_target_atom_type_gating():
    assert _last_layer(
        _model(readout_type={"atom_type_gating": {TARGET: "one-hot"}})
    ).gated
    # targets that are not listed use the default
    assert not _last_layer(
        _model(readout_type={"atom_type_gating": {"mtt::other": "one-hot"}})
    ).gated


def _check_options(hypers):
    """Merge options set by users with the defaults, as ``mtt train`` does, and
    validate them."""
    options = OmegaConf.merge(
        {"name": "pet", **get_default_hypers("pet")}, {"model": hypers}
    )
    check_architecture_options("pet", OmegaConf.to_container(options))


def test_per_target_options_are_valid():
    _check_options({"readout_type": {"atom_type_gating": {TARGET: "one-hot"}}})


def test_invalid_options():
    with pytest.raises(MetatrainValidationError):
        _check_options({"head_type": "per_atom"})


def test_v17_checkpoint_upgrade():
    """A checkpoint from before these hypers loads into the default architecture, with
    all of its weights."""
    with gzip.open("checkpoints/model-v17_trainer-v15.ckpt.gz", "rb") as fd:
        checkpoint = torch.load(fd, weights_only=False)
    state_dict = checkpoint["core"]["best_model_state_dict"]

    model = model_from_checkpoint(checkpoint, "export").core

    assert model.hypers["head_type"] == "per_target"
    assert model.hypers["num_head_layers"] == 2
    assert model.hypers["readout_type"] == {"atom_type_gating": False}
    loaded = model.state_dict()
    tensors = {k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
    assert tensors.keys() == loaded.keys()
    for name, value in tensors.items():
        torch.testing.assert_close(loaded[name], value, rtol=0, atol=0)
