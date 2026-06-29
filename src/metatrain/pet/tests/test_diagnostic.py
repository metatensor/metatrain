"""
Tests for the diagnostic hidden-layer output functionality of the PET model.

Users can request the output of any named sub-module by passing a key of the form
``"mtt::feature::<module_path>"`` in the ``outputs`` dict.  These tests verify that:

* Outputs are returned for valid module paths.
* Node-like (2-D) tensors carry per-atom sample labels.
* Edge-like (3-D) tensors carry per-pair sample labels.
* Raw featurizer input tensors (edge_vectors, etc.) are also capturable.
* GNN layers with ``_node`` / ``_edge`` suffixes work correctly.
* An informative error is raised for invalid module paths.
* Requesting diagnostic outputs does not break normal energy prediction.
"""

import pytest
import torch
from metatomic.torch import ModelOutput, System

from metatrain.pet import PET
from metatrain.pet.core import PETCore
from metatrain.pet.modules.diagnostic import (
    DIAGNOSTIC_PREFIX,
    EXCLUDED_MODULE_PREFIXES,
    FEATURIZER_INPUT_NAMES,
)
from metatrain.pet.modules.transformer import (
    CartesianTransformer,
    Transformer,
    TransformerLayer,
)
from metatrain.pet.modules.utilities import DummyModule
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from . import MODEL_HYPERS


# Module types whose forward() returns (node_tensor, edge_tensor) — these
# require a "_node" or "_edge" suffix when requesting as diagnostic outputs.
_TUPLE_MODULE_TYPES = (CartesianTransformer, Transformer, TransformerLayer)

# Container modules and placeholder modules whose forward hooks never fire:
# ModuleList/ModuleDict are called via their children, not directly; DummyModule
# raises RuntimeError if called and exists only for TorchScript compatibility.
# nn.Identity is also skipped because (a) in SwiGLU mode FeedForward registers
# an Identity as ``self.activation`` for TorchScript compatibility but never
# invokes it in forward(), so its hook would never fire; and (b) the other
# Identity instances in the model (gnn_layers_post_mp_node, node_backbone, …)
# are simple pass-throughs that do not produce interesting diagnostic tensors.
# PETCore is the pure-PyTorch core container: ``PET.forward`` calls its
# ``preprocess`` / ``compute_features`` / ``predict`` methods directly rather than
# its ``forward``, so a hook on the container itself never fires.
_SKIP_MODULE_TYPES = (
    torch.nn.ModuleList,
    torch.nn.ModuleDict,
    torch.nn.Identity,
    PETCore,
    DummyModule,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dataset_info(atomic_types=None):
    if atomic_types is None:
        atomic_types = [1, 6, 7, 8]
    return DatasetInfo(
        length_unit="Angstrom",
        atomic_types=atomic_types,
        targets={
            "energy": get_energy_target_info(
                "energy", {"quantity": "energy", "unit": "eV"}
            )
        },
    )


def _make_water_system(model):
    """Return a water molecule System with neighbor lists."""
    system = System(
        types=torch.tensor([8, 1, 1]),
        positions=torch.tensor(
            [[0.0, 0.0, 0.119], [0.0, 0.757, -0.477], [0.0, -0.757, -0.477]]
        ),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )
    return get_system_with_neighbor_lists(system, model.requested_neighbor_lists())


def _make_multi_system(model):
    """Return two water molecules as a batch (list of two Systems)."""
    return [_make_water_system(model), _make_water_system(model)]


# ---------------------------------------------------------------------------
# Basic smoke test: energy still works after hook changes
# ---------------------------------------------------------------------------


def test_energy_unaffected_by_hook_changes():
    """Requesting diagnostic outputs must not alter energy predictions."""
    dataset_info = _make_dataset_info()
    model = PET(MODEL_HYPERS, dataset_info)
    system = _make_water_system(model)

    energy_only = model([system], {"energy": ModelOutput(sample_kind="system")})
    energy_with_diag = model(
        [system],
        {
            "energy": ModelOutput(sample_kind="system"),
            "mtt::feature::core.node_heads.energy.0": ModelOutput(sample_kind="atom"),
        },
    )

    torch.testing.assert_close(
        energy_only["energy"].block().values,
        energy_with_diag["energy"].block().values,
        atol=0.0,
        rtol=0.0,
    )


# ---------------------------------------------------------------------------
# node_heads output (2-D, per-atom)
# ---------------------------------------------------------------------------


def test_node_head_output_returned():
    """mtt::feature::core.node_heads.energy.0 is present and non-empty."""
    dataset_info = _make_dataset_info()
    model = PET(MODEL_HYPERS, dataset_info)
    system = _make_water_system(model)

    outputs = {
        "mtt::feature::core.node_heads.energy.0": ModelOutput(sample_kind="atom")
    }
    result = model([system], outputs)

    assert "mtt::feature::core.node_heads.energy.0" in result
    block = result["mtt::feature::core.node_heads.energy.0"].block()
    assert block.values.shape[0] == 3  # 3 atoms in water


def test_node_head_output_sample_labels():
    """node_head output must carry per-atom (system, atom) sample labels."""
    dataset_info = _make_dataset_info()
    model = PET(MODEL_HYPERS, dataset_info)
    system = _make_water_system(model)

    outputs = {
        "mtt::feature::core.node_heads.energy.0": ModelOutput(sample_kind="atom")
    }
    result = model([system], outputs)

    block = result["mtt::feature::core.node_heads.energy.0"].block()
    assert set(block.samples.names) == {"system", "atom"}
    # One row per atom
    assert block.samples.values.shape[0] == 3


def test_node_head_output_sample_labels_batch():
    """Sample labels must cover atoms from all systems in the batch."""
    dataset_info = _make_dataset_info()
    model = PET(MODEL_HYPERS, dataset_info)
    systems = _make_multi_system(model)

    outputs = {
        "mtt::feature::core.node_heads.energy.0": ModelOutput(sample_kind="atom")
    }
    result = model(systems, outputs)

    block = result["mtt::feature::core.node_heads.energy.0"].block()
    # Two water molecules, 3 atoms each = 6 atoms total
    assert block.samples.values.shape[0] == 6
    system_col = block.samples.column("system")
    assert set(system_col.tolist()) == {0, 1}


# ---------------------------------------------------------------------------
# edge_heads output (3-D, per-pair)
# ---------------------------------------------------------------------------


def test_edge_head_output_returned():
    """mtt::feature::core.edge_heads.energy.0 is present and non-empty."""
    dataset_info = _make_dataset_info()
    model = PET(MODEL_HYPERS, dataset_info)
    system = _make_water_system(model)

    outputs = {
        "mtt::feature::core.edge_heads.energy.0": ModelOutput(sample_kind="atom")
    }
    result = model([system], outputs)

    assert "mtt::feature::core.edge_heads.energy.0" in result
    block = result["mtt::feature::core.edge_heads.energy.0"].block()
    # At least one pair must exist for a water molecule
    assert block.values.shape[0] > 0


def test_edge_head_output_sample_labels():
    """edge_head output must carry per-pair neighbor-list labels including cell
    shifts."""
    dataset_info = _make_dataset_info()
    model = PET(MODEL_HYPERS, dataset_info)
    system = _make_water_system(model)

    outputs = {
        "mtt::feature::core.edge_heads.energy.0": ModelOutput(sample_kind="atom")
    }
    result = model([system], outputs)

    block = result["mtt::feature::core.edge_heads.energy.0"].block()
    assert set(block.samples.names) == {
        "system",
        "first_atom",
        "second_atom",
        "cell_shift_a",
        "cell_shift_b",
        "cell_shift_c",
    }
    # first_atom and second_atom must be different for every pair
    first = block.samples.column("first_atom")
    second = block.samples.column("second_atom")
    assert not torch.all(first == second), "All pairs have the same center and neighbor"


def test_edge_head_pair_labels_valid_atoms():
    """Pair labels must reference atom indices that exist in the system."""
    dataset_info = _make_dataset_info()
    model = PET(MODEL_HYPERS, dataset_info)
    system = _make_water_system(model)

    outputs = {
        "mtt::feature::core.edge_heads.energy.0": ModelOutput(sample_kind="atom")
    }
    result = model([system], outputs)

    block = result["mtt::feature::core.edge_heads.energy.0"].block()
    n_atoms = len(system.types)
    first = block.samples.column("first_atom")
    second = block.samples.column("second_atom")
    assert torch.all(first >= 0) and torch.all(first < n_atoms)
    assert torch.all(second >= 0) and torch.all(second < n_atoms)


# ---------------------------------------------------------------------------
# GNN layer outputs with _node / _edge suffixes
# ---------------------------------------------------------------------------


def test_gnn_layer_node_output():
    """gnn_layers.0_node returns per-atom features."""
    dataset_info = _make_dataset_info()
    model = PET(MODEL_HYPERS, dataset_info)
    system = _make_water_system(model)

    outputs = {"mtt::feature::core.gnn_layers.0_node": ModelOutput(sample_kind="atom")}
    result = model([system], outputs)

    assert "mtt::feature::core.gnn_layers.0_node" in result
    block = result["mtt::feature::core.gnn_layers.0_node"].block()
    assert set(block.samples.names) == {"system", "atom"}
    assert block.samples.values.shape[0] == 3


def test_gnn_layer_edge_output():
    """gnn_layers.0_edge returns per-pair features."""
    dataset_info = _make_dataset_info()
    model = PET(MODEL_HYPERS, dataset_info)
    system = _make_water_system(model)

    outputs = {"mtt::feature::core.gnn_layers.0_edge": ModelOutput(sample_kind="atom")}
    result = model([system], outputs)

    assert "mtt::feature::core.gnn_layers.0_edge" in result
    block = result["mtt::feature::core.gnn_layers.0_edge"].block()
    assert set(block.samples.names) == {
        "system",
        "first_atom",
        "second_atom",
        "cell_shift_a",
        "cell_shift_b",
        "cell_shift_c",
    }
    assert block.values.shape[0] > 0


# ---------------------------------------------------------------------------
# Raw featurizer input tensors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "input_name",
    [
        "edge_vectors",
        "padding_mask",
        "element_indices_neighbors",
        "cutoff_factors",
    ],
)
def test_featurizer_input_capture(input_name):
    """Raw featurizer inputs can be requested as diagnostic outputs."""
    dataset_info = _make_dataset_info()
    model = PET(MODEL_HYPERS, dataset_info)
    system = _make_water_system(model)

    key = f"mtt::feature::{input_name}"
    result = model([system], {key: ModelOutput(sample_kind="atom")})

    assert key in result
    assert result[key].block().values.shape[0] > 0


# ---------------------------------------------------------------------------
# Node embedder (2-D output)
# ---------------------------------------------------------------------------


def test_node_embedder_output():
    """node_embedders.0 returns per-atom features."""
    dataset_info = _make_dataset_info()
    model = PET(MODEL_HYPERS, dataset_info)
    system = _make_water_system(model)

    outputs = {"mtt::feature::core.node_embedders.0": ModelOutput(sample_kind="atom")}
    result = model([system], outputs)

    assert "mtt::feature::core.node_embedders.0" in result
    block = result["mtt::feature::core.node_embedders.0"].block()
    assert set(block.samples.names) == {"system", "atom"}
    assert block.samples.values.shape[0] == 3


# ---------------------------------------------------------------------------
# Multiple diagnostic outputs in a single forward pass
# ---------------------------------------------------------------------------


def test_multiple_diagnostic_outputs():
    """Requesting multiple diagnostic outputs at once should work."""
    dataset_info = _make_dataset_info()
    model = PET(MODEL_HYPERS, dataset_info)
    system = _make_water_system(model)

    outputs = {
        "energy": ModelOutput(sample_kind="system"),
        "mtt::feature::core.node_heads.energy.0": ModelOutput(sample_kind="atom"),
        "mtt::feature::core.edge_heads.energy.0": ModelOutput(sample_kind="atom"),
        "mtt::feature::core.gnn_layers.0_node": ModelOutput(sample_kind="atom"),
    }
    result = model([system], outputs)

    assert "energy" in result
    assert "mtt::feature::core.node_heads.energy.0" in result
    assert "mtt::feature::core.edge_heads.energy.0" in result
    assert "mtt::feature::core.gnn_layers.0_node" in result


@pytest.mark.parametrize("featurizer_type", ["feedforward", "residual"])
def test_multiple_diagnostic_outputs_raw_features(featurizer_type):
    """Requesting multiple diagnostic outputs at once should work."""
    dataset_info = _make_dataset_info()
    # Modify a few of the default hypers
    hypers = MODEL_HYPERS.copy()
    hypers["d_node"] = 128
    hypers["d_pet"] = 64
    hypers["d_head"] = 32
    hypers["featurizer_type"] = featurizer_type

    model = PET(hypers, dataset_info)
    system = _make_water_system(model)

    for readout_layer in range(model.num_readout_layers):
        outputs = [
            f"mtt::feature::core.node_backbone.{readout_layer}",
            f"mtt::feature::core.edge_backbone.{readout_layer}",
            f"mtt::feature::core.node_heads.energy.{readout_layer}",
            f"mtt::feature::core.edge_heads.energy.{readout_layer}",
        ]
        result = model(
            [system], {name: ModelOutput(sample_kind="atom") for name in outputs}
        )

        assert result[outputs[0]].block().values.shape[1] == 128  # d_node
        assert result[outputs[1]].block().values.shape[1] == 64  # d_pet
        assert result[outputs[2]].block().values.shape[1] == 32  # d_head
        assert result[outputs[3]].block().values.shape[1] == 32  # d_head


# ---------------------------------------------------------------------------
# Invalid path raises informative error
# ---------------------------------------------------------------------------


def test_invalid_module_path_raises():
    """Requesting a non-existent module path must raise AttributeError."""
    dataset_info = _make_dataset_info()
    model = PET(MODEL_HYPERS, dataset_info)
    system = _make_water_system(model)

    with pytest.raises(AttributeError, match="not found in the model"):
        model(
            [system],
            {
                "mtt::feature::this_module_does_not_exist": ModelOutput(
                    sample_kind="atom"
                )
            },
        )


def test_tuple_module_without_suffix_raises():
    """
    Requesting a tuple-returning module without _node/_edge suffix must raise
    ValueError on hook invocation (not silently return wrong data).
    """
    dataset_info = _make_dataset_info()
    model = PET(MODEL_HYPERS, dataset_info)
    system = _make_water_system(model)

    with pytest.raises(ValueError, match="_node.*_edge|_edge.*_node"):
        model(
            [system],
            {"mtt::feature::core.gnn_layers.0": ModelOutput(sample_kind="atom")},
        )


# ---------------------------------------------------------------------------
# Reproducibility: two identical forward passes yield identical outputs
# ---------------------------------------------------------------------------


def test_diagnostic_outputs_are_deterministic():
    """The same forward pass should produce the same diagnostic tensors."""
    dataset_info = _make_dataset_info()
    model = PET(MODEL_HYPERS, dataset_info)
    model.eval()
    system = _make_water_system(model)

    key = "mtt::feature::core.node_heads.energy.0"
    outputs = {key: ModelOutput(sample_kind="atom")}

    result1 = model([system], outputs)
    result2 = model([system], outputs)

    torch.testing.assert_close(
        result1[key].block().values,
        result2[key].block().values,
    )


# ---------------------------------------------------------------------------
# Comprehensive sweep: every capturable module path, diatomic system
# ---------------------------------------------------------------------------


def _build_all_outputs(model) -> dict:
    """Return an ``outputs`` dict requesting every capturable module path.

    Modules whose ``forward()`` returns a ``(node_tensor, edge_tensor)`` tuple
    are requested with both the ``_node`` and ``_edge`` suffix variants.
    All raw featurizer inputs are also included.
    """
    # These module prefixes are only executed when the corresponding target
    # output (e.g. "energy") is also requested.  Including them in a
    # diagnostic-only sweep would register hooks that never fire, causing the
    # "key not in result" assertion to fail.
    _CONDITIONAL_MODULE_PREFIXES = [
        "core.node_last_layers",
        "core.edge_last_layers",
    ]

    outputs = {}

    for name, module in model.named_modules():
        if not name:
            continue  # skip the root module itself
        if any(name.startswith(p) for p in EXCLUDED_MODULE_PREFIXES):
            continue
        if any(name.startswith(p) for p in _CONDITIONAL_MODULE_PREFIXES):
            continue
        # ModuleList/ModuleDict are container types — PyTorch calls their
        # children directly, so a hook registered on the container itself
        # never fires.  DummyModule is a TorchScript placeholder that raises
        # RuntimeError if called and is never actually invoked at runtime.
        if isinstance(module, _SKIP_MODULE_TYPES):
            continue
        key_base = DIAGNOSTIC_PREFIX + name
        if isinstance(module, _TUPLE_MODULE_TYPES):
            outputs[key_base + "_node"] = ModelOutput(sample_kind="atom")
            outputs[key_base + "_edge"] = ModelOutput(sample_kind="atom")
        else:
            outputs[key_base] = ModelOutput(sample_kind="atom")

    for feat_name in FEATURIZER_INPUT_NAMES:
        outputs[DIAGNOSTIC_PREFIX + feat_name] = ModelOutput(sample_kind="atom")

    return outputs


def test_all_module_outputs_diatomic():
    """Request every named sub-module output in a single forward pass.

    The system is a diatomic H₂ molecule so that ``max_neighbors == 1`` in
    PET's NEF format.  This means every edge tensor inside the model has shape
    ``(n_atoms, 1, d)``, directly exercising the ambiguous shape that Sofia
    flagged: an edge tensor whose second dimension equals 1 looks identical to a
    node tensor with a dummy neighbor axis if we rely solely on shape to decide
    which sample labels to attach.

    The test asserts:

    * Every requested key is present in the result (no silent drop).
    * Every returned TensorMap has at least one row (non-empty).
    * Outputs requested with a ``_node`` suffix carry per-atom
      ``["system", "atom"]`` labels.
    * Outputs requested with an ``_edge`` suffix carry per-pair
      ``["system", "first_atom", "second_atom", ...]`` labels.
    """
    dataset_info = _make_dataset_info(atomic_types=[1])  # hydrogen only
    model = PET(MODEL_HYPERS, dataset_info)

    # A diatomic H-H pair 1.5 Å apart — well within the default 4.5 Å cutoff.
    # With a full neighbor list every atom sees exactly one neighbor, so
    # max_neighbors == 1 throughout the forward pass.
    system = System(
        types=torch.tensor([1, 1]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]]),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )
    system = get_system_with_neighbor_lists(system, model.requested_neighbor_lists())

    outputs = _build_all_outputs(model)
    result = model([system], outputs)

    node_label_names = ["system", "atom"]
    edge_label_names = [
        "system",
        "first_atom",
        "second_atom",
        "cell_shift_a",
        "cell_shift_b",
        "cell_shift_c",
    ]

    for key in outputs:
        assert key in result, f"Missing diagnostic output: {key}"
        block = result[key].block()
        assert block.values.shape[0] > 0, f"Empty output for: {key}"

        bare = key[len(DIAGNOSTIC_PREFIX) :]
        if bare.endswith("_node"):
            assert block.samples.names == node_label_names, (
                f"{key}: expected per-atom labels {node_label_names}, "
                f"got {block.samples.names}"
            )
        elif bare.endswith("_edge"):
            assert block.samples.names == edge_label_names, (
                f"{key}: expected per-pair labels {edge_label_names}, "
                f"got {block.samples.names}"
            )
