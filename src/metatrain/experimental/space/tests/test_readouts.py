"""
Tests for the SPACE ``readout_type`` hyperparameter.

The target used throughout is a multi-block atomic-basis (spherical) target, since
that is where atom-type-conditioned readouts are meaningful: during training such a
target is densified into one block per ``o3_lambda``, with the property axis padded
to the largest per-type basis, so every atomic type shares a block and needs its own
map onto it.
"""

import copy

import pytest
import torch
from metatomic.torch import ModelOutput, System

from metatrain.experimental.space import SPACE
from metatrain.experimental.space.modules.layers import Linear
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_generic_target_info
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists
from metatrain.utils.readout import LinearReadout

from . import MODEL_HYPERS


IRREPS = {
    1: [{"o3_lambda": 0, "o3_sigma": 1, "num": 2}],
    8: [
        {"o3_lambda": 0, "o3_sigma": 1, "num": 4},
        {"o3_lambda": 1, "o3_sigma": 1, "num": 3},
    ],
}
TARGET = "mtt::basis"


def _dataset_info():
    return DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 8],
        targets={
            TARGET: get_generic_target_info(
                TARGET,
                {
                    "quantity": "",
                    "unit": "",
                    "type": {"spherical": {"irreps": IRREPS}},
                    "num_subtargets": 1,
                    "sample_kind": "atom",
                },
            )
        },
    )


def _hypers(**overrides):
    hypers = copy.deepcopy(MODEL_HYPERS)
    hypers.update(overrides)
    return hypers


def _evaluate(model):
    model.eval()
    system = System(
        types=torch.tensor([8, 1, 1]),
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.9, 0.0], [0.9, 0.0, 0.0]], dtype=torch.float32
        ),
        cell=torch.zeros(3, 3, dtype=torch.float32),
        pbc=torch.tensor([False, False, False]),
    )
    systems = [get_system_with_neighbor_lists(system, model.requested_neighbor_lists())]
    return model(systems, {TARGET: ModelOutput(per_atom=True)})[TARGET]


def _last_layers(model):
    return model.module.module.last_layers[TARGET]


def test_ungated_default_keeps_spaces_own_linear():
    """The default path must be untouched, so init and checkpoints are too."""
    model = SPACE(_hypers(), _dataset_info())
    readout = _last_layers(model)["0"]
    assert isinstance(readout, Linear)
    # ``Linear`` accepts and ignores the group index, which is what makes it
    # interchangeable with a conditioned readout at the call site.
    features = torch.randn(5, 1, readout.n_feat_in)
    torch.testing.assert_close(
        readout(features, torch.zeros(5, dtype=torch.long)), readout(features)
    )


def test_ungated_readout_matches_spaces_linear():
    """An ungated ``LinearReadout`` is the same function as SPACE's ``Linear``."""
    linear = Linear(16, 3)
    readout = LinearReadout(16, 3, ntk_parametrization=True)
    with torch.no_grad():
        readout.weight.copy_(linear.linear_layer.weight)
    assert readout.bias is None  # a bias would break equivariance

    features = torch.randn(5, 7, 16)
    torch.testing.assert_close(
        readout(features, torch.zeros(5, dtype=torch.long)), linear(features)
    )


@pytest.mark.parametrize("gating", [False, "one-hot"])
def test_forward_runs_and_shapes_are_right(gating):
    model = SPACE(_hypers(readout_type={"atom_type_gating": gating}), _dataset_info())
    prediction = _evaluate(model)

    assert len(prediction) == 3  # l=0 for H, l=0 and l=1 for O
    for key, block in prediction.items():
        o3_lambda = int(key["o3_lambda"])
        assert block.values.shape[1] == 2 * o3_lambda + 1
        assert not block.values.isnan().any()


def test_one_hot_readout_conditions_on_the_central_atom_type():
    """An atom's prediction must depend only on its own type's readout weights."""
    model = SPACE(
        _hypers(readout_type={"atom_type_gating": "one-hot"}), _dataset_info()
    )
    readout = _last_layers(model)["0"]
    assert readout.gated
    assert readout.weight.shape[0] == 2  # atomic_types=[1, 8]
    assert readout.bias is None

    prediction = _evaluate(model)
    hydrogen_before = prediction.block({"atom_type": 1, "o3_lambda": 0}).values.clone()
    oxygen_before = prediction.block({"atom_type": 8, "o3_lambda": 0}).values.clone()

    with torch.no_grad():
        readout.weight[1] += 1.0  # oxygen's weights only

    prediction = _evaluate(model)
    torch.testing.assert_close(
        prediction.block({"atom_type": 1, "o3_lambda": 0}).values, hydrogen_before
    )
    assert not torch.allclose(
        prediction.block({"atom_type": 8, "o3_lambda": 0}).values, oxygen_before
    )


def test_gated_readout_is_equivariant():
    """Conditioning on the (invariant) atom type must not break equivariance."""
    model = SPACE(
        _hypers(readout_type={"atom_type_gating": "one-hot"}), _dataset_info()
    )
    model.eval()

    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.9, 0.0], [0.9, 0.0, 0.0]], dtype=torch.float32
    )
    # A rotation by pi/2 about z, whose l=1 (o3_mu ordering: y, z, x) action is the
    # same matrix applied to the spherical components.
    angle = torch.tensor(torch.pi / 2)
    rotation = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle), 0.0],
            [torch.sin(angle), torch.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    def predict(pos):
        system = System(
            types=torch.tensor([8, 1, 1]),
            positions=pos,
            cell=torch.zeros(3, 3, dtype=torch.float32),
            pbc=torch.tensor([False, False, False]),
        )
        systems = [
            get_system_with_neighbor_lists(system, model.requested_neighbor_lists())
        ]
        return model(systems, {TARGET: ModelOutput(per_atom=True)})[TARGET]

    original = predict(positions)
    rotated = predict(positions @ rotation.T)

    # The l=0 (invariant) block must be unchanged by the rotation.
    torch.testing.assert_close(
        original.block({"atom_type": 8, "o3_lambda": 0}).values,
        rotated.block({"atom_type": 8, "o3_lambda": 0}).values,
        rtol=1e-4,
        atol=1e-4,
    )
    # The l=1 block must rotate, i.e. change but keep its norm.
    l1_original = original.block({"atom_type": 8, "o3_lambda": 1}).values
    l1_rotated = rotated.block({"atom_type": 8, "o3_lambda": 1}).values
    torch.testing.assert_close(
        l1_original.norm(dim=1), l1_rotated.norm(dim=1), rtol=1e-4, atol=1e-4
    )


def test_per_target_readout_dict():
    model = SPACE(
        _hypers(readout_type={TARGET: {"atom_type_gating": "one-hot"}}),
        _dataset_info(),
    )
    assert isinstance(_last_layers(model)["0"], LinearReadout)

    # A target absent from the dict falls back to the default.
    model = SPACE(
        _hypers(readout_type={"mtt::other": {"atom_type_gating": "one-hot"}}),
        _dataset_info(),
    )
    assert isinstance(_last_layers(model)["0"], Linear)


def test_moe_readout():
    model = SPACE(
        _hypers(
            readout_type={
                "atom_type_gating": "moe",
                "hypers": {
                    "num_experts": 3,
                    "num_routed_experts": 2,
                    "num_topk_experts": 1,
                },
            }
        ),
        _dataset_info(),
    )
    readout = _last_layers(model)["0"]
    assert len(readout.routed_experts) == 2
    assert len(readout.shared_experts) == 1
    _evaluate(model)


def test_unknown_gating_raises():
    with pytest.raises(ValueError, match="Unknown atom_type_gating"):
        SPACE(_hypers(readout_type={"atom_type_gating": "nonsense"}), _dataset_info())
