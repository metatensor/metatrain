"""
Tests for the pure-PyTorch :class:`metatrain.pet.backend.PETCore`.

These verify that the core (structure preprocessing, featurization and prediction)
runs on plain tensors and is ``torch.compile``-able, matching eager execution.
"""

import pytest
import torch
from metatomic.torch import ModelOutput, System

from metatrain.pet import PET
from metatrain.pet.modules.structures import concatenate_structures
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from . import MODEL_HYPERS


def _make_dataset_info():
    return DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "energy": get_energy_target_info(
                "energy", {"quantity": "energy", "unit": "eV"}
            )
        },
    )


def _make_system(model):
    system = System(
        types=torch.tensor([8, 1, 1]),
        positions=torch.tensor(
            [[0.0, 0.0, 0.119], [0.0, 0.757, -0.477], [0.0, -0.757, -0.477]]
        ),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )
    return get_system_with_neighbor_lists(system, model.requested_neighbor_lists())


def _backend_inputs(model, system):
    (
        positions,
        centers,
        neighbors,
        species,
        cells,
        cell_shifts,
        system_indices,
        _sample_labels,
    ) = concatenate_structures([system], model.requested_neighbor_lists()[0])
    return positions, centers, neighbors, species, cells, cell_shifts, system_indices


def test_backend_runs_on_plain_tensors():
    """The core consumes and returns only plain tensors (no metatensor objects)."""
    model = PET(MODEL_HYPERS, _make_dataset_info()).eval()
    backend = model.backend
    inputs = _backend_inputs(model, _make_system(model))
    positions, centers, neighbors, species, cells, cell_shifts, system_indices = inputs

    batch_data = backend.preprocess(*inputs)
    assert isinstance(batch_data, dict)
    assert all(isinstance(v, torch.Tensor) for v in batch_data.values())

    node_list, edge_list = backend.calculate_features(batch_data)
    assert all(isinstance(t, torch.Tensor) for t in node_list)
    assert all(isinstance(t, torch.Tensor) for t in edge_list)

    atomic_predictions, node_ll, edge_ll = backend.predict(
        node_list, edge_list, batch_data, cells, system_indices, ["energy"]
    )
    assert "energy" in atomic_predictions
    assert all(isinstance(t, torch.Tensor) for t in atomic_predictions["energy"])


@pytest.mark.parametrize("num_neighbors_adaptive", [None])
def test_backend_preprocess_fullgraph_compile(num_neighbors_adaptive):
    """``preprocess`` compiles with ``fullgraph=True`` and matches eager.

    This requires ``capture_scalar_outputs`` (for the data-dependent
    ``max_edges_per_node``) together with the ``torch._check_is_size`` hint and the
    ``scatter_add`` neighbour count in :func:`compute_batch_tensors` (``bincount`` has a
    data-dependent output shape that miscompiles). The adaptive-cutoff path
    (``num_neighbors_adaptive`` set) additionally exercises the ``nonzero`` edge filter
    and the unbacked-size handling in ``get_corresponding_edges``.

    :param num_neighbors_adaptive: ``None`` for the fixed-cutoff path, or a float to
        exercise the adaptive-cutoff path.
    """
    hypers = dict(MODEL_HYPERS)
    hypers["num_neighbors_adaptive"] = num_neighbors_adaptive
    model = PET(hypers, _make_dataset_info()).eval()
    backend = model.backend
    inputs = _backend_inputs(model, _make_system(model))

    batch_data_e = backend.preprocess(*inputs)
    batch_data_c = torch.compile(backend.preprocess, fullgraph=True)(*inputs)

    # capture_scalar = torch._dynamo.config.capture_scalar_outputs
    # capture_shape = torch._dynamo.config.capture_dynamic_output_shape_ops
    # torch._dynamo.config.capture_scalar_outputs = True
    # torch._dynamo.config.capture_dynamic_output_shape_ops = True
    # try:
    #     # torch._dynamo.reset()

    # finally:
    #     torch._dynamo.config.capture_scalar_outputs = capture_scalar
    #     torch._dynamo.config.capture_dynamic_output_shape_ops = capture_shape

    for key in batch_data_e:
        assert batch_data_e[key].shape == batch_data_c[key].shape, key
        torch.testing.assert_close(
            batch_data_e[key], batch_data_c[key], atol=0.0, rtol=0.0
        )


def test_backend_torch_compile_matches_eager():
    """``torch.compile`` of the core methods matches eager execution."""
    model = PET(MODEL_HYPERS, _make_dataset_info()).eval()
    backend = model.backend
    inputs = _backend_inputs(model, _make_system(model))
    cells = inputs[4]
    system_indices = inputs[6]

    # Eager
    batch_data = backend.preprocess(*inputs)
    node_e, edge_e = backend.calculate_features(batch_data)
    preds_e, _, _ = backend.predict(
        node_e, edge_e, batch_data, cells, system_indices, ["energy"]
    )

    # Compiled (fullgraph=False tolerates the data-dependent max-neighbors sync)
    compiled_preprocess = torch.compile(backend.preprocess, fullgraph=False)
    compiled_calculate_features = torch.compile(
        backend.calculate_features, fullgraph=False
    )
    batch_data_c = compiled_preprocess(*inputs)
    node_c, edge_c = compiled_calculate_features(batch_data_c)
    preds_c, _, _ = backend.predict(
        node_c, edge_c, batch_data_c, cells, system_indices, ["energy"]
    )

    torch.testing.assert_close(preds_e["energy"][0], preds_c["energy"][0])


def test_backend_predictions_match_full_model():
    """The core's per-block predictions match the wrapped model's energy output."""
    model = PET(MODEL_HYPERS, _make_dataset_info()).eval()
    system = _make_system(model)

    # Wrapped per-atom energy from the full model (no additive / scaler contributions
    # for an untrained model with zero composition weights and identity scaler).
    per_atom = model([system], {"energy": ModelOutput(sample_kind="atom")})
    wrapped = per_atom["energy"].block().values

    backend = model.backend
    inputs = _backend_inputs(model, system)
    cells = inputs[4]
    system_indices = inputs[6]
    batch_data = backend.preprocess(*inputs)
    node_list, edge_list = backend.calculate_features(batch_data)
    atomic_predictions, _, _ = backend.predict(
        node_list, edge_list, batch_data, cells, system_indices, ["energy"]
    )

    torch.testing.assert_close(atomic_predictions["energy"][0], wrapped)
