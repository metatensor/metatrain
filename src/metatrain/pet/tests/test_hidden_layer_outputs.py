"""
Tests for the diagnostic hidden-layer output functionality of the PET model.

Users can request the output of any named sub-module by passing a key of the form
``"mtt::features::<module_path>"`` in the ``outputs`` dict.  These tests verify that:

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
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from . import MODEL_HYPERS


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

    energy_only = model([system], {"energy": ModelOutput(per_atom=False)})
    energy_with_diag = model(
        [system],
        {
            "energy": ModelOutput(per_atom=False),
            "mtt::features::node_heads.energy.0": ModelOutput(per_atom=True),
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
    """mtt::features::node_heads.energy.0 is present and non-empty."""
    dataset_info = _make_dataset_info()
    model = PET(MODEL_HYPERS, dataset_info)
    system = _make_water_system(model)

    outputs = {"mtt::features::node_heads.energy.0": ModelOutput(per_atom=True)}
    result = model([system], outputs)

    assert "mtt::features::node_heads.energy.0" in result
    block = result["mtt::features::node_heads.energy.0"].block()
    assert block.values.shape[0] == 3  # 3 atoms in water


def test_node_head_output_sample_labels():
    """node_head output must carry per-atom (system, atom) sample labels."""
    dataset_info = _make_dataset_info()
    model = PET(MODEL_HYPERS, dataset_info)
    system = _make_water_system(model)

    outputs = {"mtt::features::node_heads.energy.0": ModelOutput(per_atom=True)}
    result = model([system], outputs)

    block = result["mtt::features::node_heads.energy.0"].block()
    assert set(block.samples.names) == {"system", "atom"}
    # One row per atom
    assert block.samples.values.shape[0] == 3


def test_node_head_output_sample_labels_batch():
    """Sample labels must cover atoms from all systems in the batch."""
    dataset_info = _make_dataset_info()
    model = PET(MODEL_HYPERS, dataset_info)
    systems = _make_multi_system(model)

    outputs = {"mtt::features::node_heads.energy.0": ModelOutput(per_atom=True)}
    result = model(systems, outputs)

    block = result["mtt::features::node_heads.energy.0"].block()
    # Two water molecules, 3 atoms each = 6 atoms total
    assert block.samples.values.shape[0] == 6
    system_col = block.samples.column("system")
    assert set(system_col.tolist()) == {0, 1}


# ---------------------------------------------------------------------------
# edge_heads output (3-D, per-pair)
# ---------------------------------------------------------------------------


def test_edge_head_output_returned():
    """mtt::features::edge_heads.energy.0 is present and non-empty."""
    dataset_info = _make_dataset_info()
    model = PET(MODEL_HYPERS, dataset_info)
    system = _make_water_system(model)

    outputs = {"mtt::features::edge_heads.energy.0": ModelOutput(per_atom=True)}
    result = model([system], outputs)

    assert "mtt::features::edge_heads.energy.0" in result
    block = result["mtt::features::edge_heads.energy.0"].block()
    # At least one pair must exist for a water molecule
    assert block.values.shape[0] > 0


def test_edge_head_output_sample_labels():
    """edge_head output must carry per-pair neighbor-list labels including cell
    shifts."""
    dataset_info = _make_dataset_info()
    model = PET(MODEL_HYPERS, dataset_info)
    system = _make_water_system(model)

    outputs = {"mtt::features::edge_heads.energy.0": ModelOutput(per_atom=True)}
    result = model([system], outputs)

    block = result["mtt::features::edge_heads.energy.0"].block()
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

    outputs = {"mtt::features::edge_heads.energy.0": ModelOutput(per_atom=True)}
    result = model([system], outputs)

    block = result["mtt::features::edge_heads.energy.0"].block()
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

    outputs = {"mtt::features::gnn_layers.0_node": ModelOutput(per_atom=True)}
    result = model([system], outputs)

    assert "mtt::features::gnn_layers.0_node" in result
    block = result["mtt::features::gnn_layers.0_node"].block()
    assert set(block.samples.names) == {"system", "atom"}
    assert block.samples.values.shape[0] == 3


def test_gnn_layer_edge_output():
    """gnn_layers.0_edge returns per-pair features."""
    dataset_info = _make_dataset_info()
    model = PET(MODEL_HYPERS, dataset_info)
    system = _make_water_system(model)

    outputs = {"mtt::features::gnn_layers.0_edge": ModelOutput(per_atom=True)}
    result = model([system], outputs)

    assert "mtt::features::gnn_layers.0_edge" in result
    block = result["mtt::features::gnn_layers.0_edge"].block()
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

    key = f"mtt::features::{input_name}"
    result = model([system], {key: ModelOutput(per_atom=True)})

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

    outputs = {"mtt::features::node_embedders.0": ModelOutput(per_atom=True)}
    result = model([system], outputs)

    assert "mtt::features::node_embedders.0" in result
    block = result["mtt::features::node_embedders.0"].block()
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
        "energy": ModelOutput(per_atom=False),
        "mtt::features::node_heads.energy.0": ModelOutput(per_atom=True),
        "mtt::features::edge_heads.energy.0": ModelOutput(per_atom=True),
        "mtt::features::gnn_layers.0_node": ModelOutput(per_atom=True),
    }
    result = model([system], outputs)

    assert "energy" in result
    assert "mtt::features::node_heads.energy.0" in result
    assert "mtt::features::edge_heads.energy.0" in result
    assert "mtt::features::gnn_layers.0_node" in result


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
            f"mtt::features::node_backbone.{readout_layer}",
            f"mtt::features::edge_backbone.{readout_layer}",
            f"mtt::features::node_heads.energy.{readout_layer}",
            f"mtt::features::edge_heads.energy.{readout_layer}",
        ]
        result = model([system], {name: ModelOutput(per_atom=True) for name in outputs})

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
            {"mtt::features::this_module_does_not_exist": ModelOutput(per_atom=True)},
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
            {"mtt::features::gnn_layers.0": ModelOutput(per_atom=True)},
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

    key = "mtt::features::node_heads.energy.0"
    outputs = {key: ModelOutput(per_atom=True)}

    result1 = model([system], outputs)
    result2 = model([system], outputs)

    torch.testing.assert_close(
        result1[key].block().values,
        result2[key].block().values,
    )
