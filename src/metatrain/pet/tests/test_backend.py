"""
Tests for the pure-PyTorch :class:`metatrain.pet.backend.PETBackend`.

These verify that the backend (structure preprocessing, featurization and prediction)
runs on plain tensors and is ``torch.compile``-able, matching eager execution.
"""

import contextlib
import warnings

import metatensor.torch as mts
import pytest
import torch
from metatomic.torch import ModelOutput, System, register_autograd_neighbors

from metatrain.pet import PET
from metatrain.pet.modules.structures import concatenate_structures
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from . import MODEL_HYPERS


@contextlib.contextmanager
def _ignore_nonleaf_grad_warning():
    """Silence the benign non-leaf ``.grad`` warning while compiling ``predict``.

    When Dynamo's builder wraps a grad-tracking (non-leaf, ``requires_grad``) tensor as
    a graph input, it reads its ``.grad`` and PyTorch emits a harmless ``UserWarning``.
    The repo's ``filterwarnings = ["error", ...]`` pytest config would escalate that to
    an exception (surfaced as ``InternalTorchDynamoError``), so we ignore just this one
    message. The autograd graph is left intact, so forces via ``autograd.grad`` still
    work.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*grad attribute of a Tensor that is not a leaf Tensor.*",
            category=UserWarning,
        )
        yield


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
            [[0.0, 0.0, 0.119], [0.0, 0.757, -0.477], [0.0, -0.757, -0.477]],
        ),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )
    return get_system_with_neighbor_lists(system, model.requested_neighbor_lists())


def _make_periodic_system(model):
    """A small periodic system so that strain (stress) gradients are non-trivial.

    A 3.5 Angstrom cubic cell with a 4.5 Angstrom cutoff pulls in periodic images, so
    the ``cell_shifts @ cells`` term in the edge vectors carries a real strain gradient.
    """
    system = System(
        types=torch.tensor([6, 8]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [1.5, 1.5, 1.5]]),
        cell=3.5 * torch.eye(3),
        pbc=torch.tensor([True, True, True]),
    )
    return get_system_with_neighbor_lists(system, model.requested_neighbor_lists())


def _forces_and_strain_grad(energy_fn, base_positions):
    """Compute ``(forces, strain_grad)`` from an energy closure via the strain trick.

    ``energy_fn`` maps ``(positions, strain)`` to a scalar energy; ``positions`` and
    ``strain`` are fresh leaf tensors so the returned ``forces = -dE/dpositions`` and
    ``strain_grad = dE/dstrain`` come straight from :func:`torch.autograd.grad`.
    """
    positions = base_positions.detach().clone().requires_grad_(True)
    strain = torch.eye(3, dtype=base_positions.dtype).requires_grad_(True)
    energy = energy_fn(positions, strain)
    minus_forces, strain_grad = torch.autograd.grad(energy, [positions, strain])
    return -minus_forces, strain_grad


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
    """The backend consumes and returns only plain tensors (no metatensor objects)."""
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


@pytest.mark.parametrize("num_neighbors_adaptive", [None, 5.0])
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
    with torch._dynamo.config.patch(
        capture_scalar_outputs=True,
        capture_dynamic_output_shape_ops=True,
        specialize_int=True,
    ):
        batch_data_c = torch.compile(backend.preprocess, fullgraph=True)(*inputs)

    for key in batch_data_e:
        assert batch_data_e[key].shape == batch_data_c[key].shape, key
        torch.testing.assert_close(
            batch_data_e[key], batch_data_c[key], atol=0.0, rtol=0.0
        )


def test_backend_torch_compile_matches_eager():
    """``torch.compile`` of the backend methods matches eager execution."""
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

    with torch._dynamo.config.patch(
        capture_scalar_outputs=True,
        capture_dynamic_output_shape_ops=True,
        specialize_int=True,
    ):
        compiled_preprocess = torch.compile(backend.preprocess, fullgraph=True)
        compiled_calculate_features = torch.compile(
            backend.calculate_features, fullgraph=True
        )
        compiled_predict = torch.compile(backend.predict, fullgraph=True)

    with _ignore_nonleaf_grad_warning():
        batch_data_c = compiled_preprocess(*inputs)
        node_c, edge_c = compiled_calculate_features(batch_data_c)
        preds_c, _, _ = compiled_predict(
            node_c, edge_c, batch_data_c, cells, system_indices, ["energy"]
        )

    torch.testing.assert_close(preds_e["energy"][0], preds_c["energy"][0])


def test_backend_predictions_match_full_model():
    """The backend's per-block predictions match the wrapped model's energy output."""
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


def test_compiled_backend_predictions_match_full_model():
    """``torch.compile`` of the backend's methods matches the wrapped energy output."""
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
    # The Dynamo flags must be active while the compiled functions actually trace,
    # which happens lazily on first *call* — so the calls (not just the
    # ``torch.compile`` wrapping) must run inside the ``config.patch`` context. The
    # autograd graph is kept intact (no ``no_grad``); the warning filter only silences
    # the benign non-leaf ``.grad`` warning so forces via autograd remain possible.
    with torch._dynamo.config.patch(
        capture_scalar_outputs=True,
        capture_dynamic_output_shape_ops=True,
        specialize_int=True,
    ):
        backend.preprocess = torch.compile(backend.preprocess, fullgraph=True)
        backend.calculate_features = torch.compile(
            backend.calculate_features, fullgraph=True
        )
        backend.predict = torch.compile(backend.predict, fullgraph=True)

    with _ignore_nonleaf_grad_warning():
        batch_data_c = backend.preprocess(*inputs)
        node_c, edge_c = backend.calculate_features(batch_data_c)
        atomic_predictions_c, _, _ = backend.predict(
            node_c, edge_c, batch_data_c, cells, system_indices, ["energy"]
        )

    torch.testing.assert_close(atomic_predictions_c["energy"][0], wrapped)


def test_compiled_backend_forces_and_stresses_match_full_model():
    """Forces and stresses from the compiled backend match the wrapped model.

    Both quantities come from autograd of the total energy via the strain trick
    (``positions @ strain``, ``cell @ strain``): ``forces = -dE/dpositions`` and the
    strain gradient ``dE/dstrain`` (the unnormalised stress / virial). A periodic
    system is used so that the ``cell_shifts @ cells`` edge term carries a real strain
    gradient, exercising the backend's cell path.
    """
    model = PET(MODEL_HYPERS, _make_dataset_info()).eval()
    system = _make_periodic_system(model)

    # Reference forces and strain gradient from the full (eager) model. This must run
    # before the backend methods are compiled, since the full model shares the same
    # backend instance.
    def full_energy(positions, strain):
        new_system = System(
            positions=positions @ strain,
            cell=system.cell @ strain,
            types=system.types,
            pbc=system.pbc,
        )
        for options in system.known_neighbor_lists():
            neighbors = mts.detach_block(system.get_neighbor_list(options))
            register_autograd_neighbors(new_system, neighbors)
            new_system.add_neighbor_list(options, neighbors)
        out = model([new_system], {"energy": ModelOutput(sample_kind="system")})
        return out["energy"].block().values.sum()

    forces_full, strain_grad_full = _forces_and_strain_grad(
        full_energy, system.positions
    )

    (
        positions,
        centers,
        neighbors,
        species,
        cells,
        cell_shifts,
        system_indices,
    ) = _backend_inputs(model, system)

    backend = model.backend

    with torch._dynamo.config.patch(
        capture_scalar_outputs=True,
        capture_dynamic_output_shape_ops=True,
        specialize_int=True,
    ):
        backend.preprocess = torch.compile(backend.preprocess, fullgraph=True)
        backend.calculate_features = torch.compile(
            backend.calculate_features, fullgraph=True
        )
        backend.predict = torch.compile(backend.predict, fullgraph=True)

    def backend_energy(positions, strain):
        strained_cells = cells @ strain
        batch_data = backend.preprocess(
            positions @ strain,
            centers,
            neighbors,
            species,
            strained_cells,
            cell_shifts,
            system_indices,
        )
        node_list, edge_list = backend.calculate_features(batch_data)
        preds, _, _ = backend.predict(
            node_list,
            edge_list,
            batch_data,
            strained_cells,
            system_indices,
            ["energy"],
        )
        return preds["energy"][0].sum()

    with _ignore_nonleaf_grad_warning():
        forces_backend, strain_grad_backend = _forces_and_strain_grad(
            backend_energy, positions
        )

    torch.testing.assert_close(forces_backend, forces_full)
    torch.testing.assert_close(strain_grad_backend, strain_grad_full)
