"""
Tests for the PET ``head_type``, ``num_head_layers``, ``d_head`` and
``readout_type`` hyperparameters.

The target used throughout is a multi-block atomic-basis (spherical) target, since
that is where per-block heads and atom-type-conditioned readouts are meaningful:
during training such a target is densified into one block per ``o3_lambda``, with
the property axis padded to the largest per-type basis, so every atomic type shares
a block and needs its own map onto it.
"""

import copy

import pytest
import torch
from metatomic.torch import ModelOutput, System

from metatrain.pet import PET
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_generic_target_info
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from . import MODEL_HYPERS


# H carries an l=0 basis only, O an l=0 and an l=1 basis: two blocks once densified.
IRREPS = {
    1: [{"o3_lambda": 0, "o3_sigma": 1, "num": 2}],
    8: [
        {"o3_lambda": 0, "o3_sigma": 1, "num": 4},
        {"o3_lambda": 1, "o3_sigma": 1, "num": 3},
    ],
}
NUM_BLOCKS = 2
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


def _systems(model):
    system = System(
        types=torch.tensor([8, 1, 1]),
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.9, 0.0], [0.9, 0.0, 0.0]], dtype=torch.float32
        ),
        cell=torch.zeros(3, 3, dtype=torch.float32),
        pbc=torch.tensor([False, False, False]),
    )
    return [get_system_with_neighbor_lists(system, model.requested_neighbor_lists())]


def _evaluate(model):
    model.eval()
    return model(_systems(model), {TARGET: ModelOutput(per_atom=True)})[TARGET]


def _block_keys(model):
    """The keys the backend uses for a target's per-block readout modules."""
    return list(model.backend.node_last_layers[TARGET][0].keys())


@pytest.mark.parametrize(
    "head_type, expected_heads_per_layer",
    [("per_target", 1), ("per_block", NUM_BLOCKS)],
)
def test_head_type_controls_number_of_heads(head_type, expected_heads_per_layer):
    """``per_block`` builds one head per block; ``per_target`` a single shared one."""
    model = PET(_hypers(head_type=head_type), _dataset_info())
    backend = model.backend

    assert backend.heads_per_layer[TARGET] == expected_heads_per_layer
    expected = backend.num_readout_layers * expected_heads_per_layer
    assert len(backend.node_heads[TARGET]) == expected
    assert len(backend.edge_heads[TARGET]) == expected
    # The readouts are per block regardless of the head type.
    assert len(backend.node_last_layers[TARGET][0]) == NUM_BLOCKS


def test_per_target_head_layout_is_unchanged():
    """The default must keep main's parameter names, so old checkpoints still load."""
    model = PET(_hypers(), _dataset_info())
    names = set(model.state_dict().keys())
    block_key = _block_keys(model)[0]
    assert f"backend.node_heads.{TARGET}.0.0.weight" in names
    # ``LinearReadout`` keeps the ``weight`` / ``bias`` names and shapes of the
    # ``torch.nn.Linear`` last layer it replaces, so old checkpoints load as-is.
    assert f"backend.node_last_layers.{TARGET}.0.{block_key}.weight" in names
    assert f"backend.node_last_layers.{TARGET}.0.{block_key}.bias" in names


@pytest.mark.parametrize("head_type", ["per_target", "per_block"])
@pytest.mark.parametrize("gating", [False, "one-hot"])
def test_forward_runs_and_shapes_are_right(head_type, gating):
    model = PET(
        _hypers(head_type=head_type, readout_type={"atom_type_gating": gating}),
        _dataset_info(),
    )
    prediction = _evaluate(model)

    # Eval mode sparsifies back to one block per (o3_lambda, o3_sigma, atom_type):
    # l=0 for H, and l=0 and l=1 for O.
    assert len(prediction) == 3
    for key, block in prediction.items():
        o3_lambda = int(key["o3_lambda"])
        assert block.values.shape[1] == 2 * o3_lambda + 1
        assert not block.values.isnan().any()


def test_per_block_heads_are_independent():
    """Each block must read from its own head under ``per_block``."""
    model = PET(_hypers(head_type="per_block"), _dataset_info())
    model.eval()
    before = [b.values.clone() for b in _evaluate(model).blocks()]

    # Perturb the head feeding the *last* block only.
    with torch.no_grad():
        for parameter in model.backend.node_heads[TARGET][NUM_BLOCKS - 1].parameters():
            parameter += 1.0
    after = [b.values for b in _evaluate(model).blocks()]

    changed = [not torch.allclose(a, b) for a, b in zip(before, after, strict=True)]
    assert any(changed), "perturbing a per-block head changed nothing"
    assert not all(changed), "every block changed; the heads are not independent"


def test_one_hot_readout_conditions_on_the_central_atom_type():
    """An atom's prediction must depend only on its own type's readout weights."""
    model = PET(_hypers(readout_type={"atom_type_gating": "one-hot"}), _dataset_info())
    model.eval()
    readout = model.backend.node_last_layers[TARGET][0][_block_keys(model)[0]]
    # atomic_types=[1, 8], so group 0 is H and group 1 is O.
    assert readout.weight.shape[0] == 2

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


def test_asymmetric_d_head():
    model = PET(
        _hypers(d_head={"node": 32, "edge": 64}, head_type="per_block"),
        _dataset_info(),
    )
    backend = model.backend
    assert backend.node_heads[TARGET][0][0].out_features == 32
    assert backend.edge_heads[TARGET][0][0].out_features == 64
    assert backend.node_last_layers[TARGET][0][_block_keys(model)[0]].in_features == 32
    assert backend.edge_last_layers[TARGET][0][_block_keys(model)[0]].in_features == 64
    _evaluate(model)


@pytest.mark.parametrize("num_head_layers", [1, 3])
def test_num_head_layers(num_head_layers):
    model = PET(_hypers(num_head_layers=num_head_layers), _dataset_info())
    head = model.backend.node_heads[TARGET][0]
    assert len(head) == 2 * num_head_layers  # Linear + SiLU per layer
    _evaluate(model)


def test_per_target_hyper_dicts():
    """``head_type`` and ``readout_type`` may be keyed by target name."""
    model = PET(
        _hypers(
            head_type={TARGET: "per_block"},
            readout_type={TARGET: {"atom_type_gating": "one-hot"}},
        ),
        _dataset_info(),
    )
    assert model.backend.heads_per_layer[TARGET] == NUM_BLOCKS
    assert model.backend.node_last_layers[TARGET][0][_block_keys(model)[0]].gated
    _evaluate(model)

    # A target absent from the dicts falls back to the defaults.
    model = PET(_hypers(head_type={"mtt::other": "per_block"}), _dataset_info())
    assert model.backend.heads_per_layer[TARGET] == 1


def test_moe_readout():
    model = PET(
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
    readout = model.backend.node_last_layers[TARGET][0][_block_keys(model)[0]]
    assert len(readout.routed_experts) == 2
    assert len(readout.shared_experts) == 1
    _evaluate(model)


def test_unknown_gating_raises():
    with pytest.raises(ValueError, match="Unknown atom_type_gating"):
        PET(_hypers(readout_type={"atom_type_gating": "nonsense"}), _dataset_info())


def test_invalid_num_head_layers_raises():
    with pytest.raises(ValueError, match="num_head_layers must be >= 1"):
        PET(_hypers(num_head_layers=0), _dataset_info())


@pytest.mark.parametrize("head_type", ["per_target", "per_block"])
@pytest.mark.parametrize("gating", [False, "one-hot", "moe"])
def test_torchscript(head_type, gating):
    """Every head/readout combination must still export."""
    readout_type = {"atom_type_gating": gating}
    if gating == "moe":
        readout_type["hypers"] = {
            "num_experts": 3,
            "num_routed_experts": 2,
            "num_topk_experts": 1,
        }
    model = PET(
        _hypers(head_type=head_type, readout_type=readout_type), _dataset_info()
    )
    torch.jit.script(model)
