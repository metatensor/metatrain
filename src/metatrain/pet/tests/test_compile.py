"""Tests for torch.compile support in PET."""

import copy

import pytest
import torch
from metatomic.torch import ModelOutput

from metatrain.pet.modules.transformer import CartesianTransformer
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.testing import ArchitectureTests, TrainingTests


class PETTests(ArchitectureTests):
    architecture = "pet"

    @pytest.fixture
    def minimal_model_hypers(self):
        hypers = get_default_hypers(self.architecture)["model"]
        hypers = copy.deepcopy(hypers)
        hypers["d_pet"] = 1
        hypers["d_head"] = 1
        hypers["d_node"] = 1
        hypers["d_feedforward"] = 1
        hypers["num_heads"] = 1
        hypers["num_attention_layers"] = 1
        hypers["num_gnn_layers"] = 1
        return hypers


def _make_cartesian_transformer(is_first=True, transformer_type="PreLN"):
    """Helper to create a test CartesianTransformer."""
    return CartesianTransformer(
        cutoff=4.5,
        cutoff_width=0.5,
        d_model=8,
        n_head=2,
        dim_node_features=16,
        dim_feedforward=8,
        n_layers=2,
        norm="RMSNorm",
        activation="SwiGLU",
        attention_temperature=1.0,
        transformer_type=transformer_type,
        n_atomic_species=4,
        is_first=is_first,
    )


def _make_inputs(n_atoms=5, max_neighbors=10, d_model=8, dim_node_features=16):
    """Helper to create test inputs for CartesianTransformer."""
    input_node_embeddings = torch.randn(n_atoms, dim_node_features)
    input_messages = torch.randn(n_atoms, max_neighbors, d_model)
    element_indices_neighbors = torch.randint(0, 4, (n_atoms, max_neighbors))
    edge_vectors = torch.randn(n_atoms, max_neighbors, 3)
    padding_mask = torch.ones(n_atoms, max_neighbors, dtype=torch.bool)
    padding_mask[:, -3:] = False
    edge_distances = torch.randn(n_atoms, max_neighbors).abs()
    cutoff_factors = torch.rand(n_atoms, max_neighbors)
    cutoff_factors[~padding_mask] = 0.0
    return (
        input_node_embeddings,
        input_messages,
        element_indices_neighbors,
        edge_vectors,
        padding_mask,
        edge_distances,
        cutoff_factors,
    )


def test_compile_cartesian_transformer():
    """Test CartesianTransformer with fullgraph=True and SDPA attention."""
    ct = _make_cartesian_transformer()
    compiled_ct = torch.compile(ct, fullgraph=True)

    inputs = _make_inputs()
    out_eager = ct(*inputs, False)
    out_compiled = compiled_ct(*inputs, False)

    assert torch.allclose(out_eager[0], out_compiled[0], atol=1e-5)
    assert torch.allclose(out_eager[1], out_compiled[1], atol=1e-5)


def test_compile_manual_attention():
    """Test that CartesianTransformer compiles with manual attention path."""
    ct = _make_cartesian_transformer()
    compiled_ct = torch.compile(ct, fullgraph=True)

    inputs = _make_inputs()
    out_eager = ct(*inputs, True)
    out_compiled = compiled_ct(*inputs, True)

    assert torch.allclose(out_eager[0], out_compiled[0], atol=1e-5)
    assert torch.allclose(out_eager[1], out_compiled[1], atol=1e-5)


def test_compile_backward():
    """Test that single backward through compiled CartesianTransformer works."""
    ct = _make_cartesian_transformer()
    compiled_ct = torch.compile(ct, fullgraph=True)

    inputs = list(_make_inputs())
    inputs[3] = inputs[3].requires_grad_(True)  # edge_vectors

    out = compiled_ct(*inputs, False)
    loss = out[0].sum() + out[1].sum()
    loss.backward()

    assert inputs[3].grad is not None
    assert inputs[3].grad.shape == inputs[3].shape


def test_compile_not_first_layer():
    """Test compilation of non-first CartesianTransformer (different forward branch)."""
    ct = _make_cartesian_transformer(is_first=False)
    compiled_ct = torch.compile(ct, fullgraph=True)

    inputs = _make_inputs()
    out_eager = ct(*inputs, False)
    out_compiled = compiled_ct(*inputs, False)

    assert torch.allclose(out_eager[0], out_compiled[0], atol=1e-5)
    assert torch.allclose(out_eager[1], out_compiled[1], atol=1e-5)


def test_compile_postln():
    """Test compilation with PostLN transformer type."""
    ct = _make_cartesian_transformer(transformer_type="PostLN")
    compiled_ct = torch.compile(ct, fullgraph=True)

    inputs = _make_inputs()
    out_eager = ct(*inputs, False)
    out_compiled = compiled_ct(*inputs, False)

    assert torch.allclose(out_eager[0], out_compiled[0], atol=1e-5)
    assert torch.allclose(out_eager[1], out_compiled[1], atol=1e-5)


def test_forward_from_batch():
    """Test that _forward_from_batch matches forward for per-atom energy."""
    from metatrain.pet import PET
    from metatrain.utils.data import DatasetInfo
    from metatrain.utils.data.readers import read_systems
    from metatrain.utils.data.target_info import get_energy_target_info
    from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

    from ..modules.structures import systems_to_batch
    from . import DATASET_PATH, MODEL_HYPERS

    torch.manual_seed(42)

    targets = {
        "mtt::U0": get_energy_target_info(
            "mtt::U0", {"quantity": "energy", "unit": "eV"}
        )
    }
    dataset_info = DatasetInfo(
        length_unit="Angstrom", atomic_types=[1, 6, 7, 8], targets=targets
    )
    model = PET(MODEL_HYPERS, dataset_info)
    model.eval()

    systems = read_systems(DATASET_PATH)[:3]
    systems = [s.to(torch.float32) for s in systems]
    for s in systems:
        get_system_with_neighbor_lists(s, model.requested_neighbor_lists())

    # Get per-atom predictions from forward
    forward_output = model(
        systems,
        {"mtt::U0": ModelOutput(quantity="energy", unit="", per_atom=True)},
    )
    forward_per_atom = forward_output["mtt::U0"].block().values

    # Get per-atom predictions from _forward_from_batch
    (
        element_indices_nodes,
        element_indices_neighbors,
        edge_vectors,
        edge_distances,
        padding_mask,
        reverse_neighbor_index,
        cutoff_factors,
        system_indices,
        neighbor_atom_indices,
        sample_labels,
    ) = systems_to_batch(
        systems,
        model.requested_nl,
        model.atomic_types,
        model.species_to_species_index,
        model.cutoff_function,
        model.cutoff_width,
        model.num_neighbors_adaptive,
    )

    batch_output = model._forward_from_batch(
        element_indices_nodes,
        element_indices_neighbors,
        edge_vectors,
        edge_distances,
        padding_mask,
        reverse_neighbor_index,
        cutoff_factors,
    )
    # Get the first (and only) block key for the energy target
    energy_key = next(iter(model.output_shapes["mtt::U0"]))
    batch_per_atom = batch_output["mtt::U0"][energy_key]

    torch.testing.assert_close(forward_per_atom, batch_per_atom, atol=1e-6, rtol=1e-6)


def test_forward_from_batch_adaptive():
    """Test that _forward_from_batch matches forward with adaptive cutoffs.

    Adaptive cutoffs (num_neighbors_adaptive=16) cause max_edges_per_node
    to vary per batch, which exercises dynamic=True more aggressively.
    """
    from metatrain.pet import PET
    from metatrain.utils.data import DatasetInfo
    from metatrain.utils.data.readers import read_systems
    from metatrain.utils.data.target_info import get_energy_target_info
    from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

    from ..modules.structures import systems_to_batch
    from . import DATASET_PATH, MODEL_HYPERS

    torch.manual_seed(42)

    hypers = copy.deepcopy(MODEL_HYPERS)
    hypers["num_neighbors_adaptive"] = 16

    targets = {
        "mtt::U0": get_energy_target_info(
            "mtt::U0", {"quantity": "energy", "unit": "eV"}
        )
    }
    dataset_info = DatasetInfo(
        length_unit="Angstrom", atomic_types=[1, 6, 7, 8], targets=targets
    )
    model = PET(hypers, dataset_info)
    model.eval()

    systems = read_systems(DATASET_PATH)[:3]
    systems = [s.to(torch.float32) for s in systems]
    for s in systems:
        get_system_with_neighbor_lists(s, model.requested_neighbor_lists())

    # Get per-atom predictions from forward
    forward_output = model(
        systems,
        {"mtt::U0": ModelOutput(quantity="energy", unit="", per_atom=True)},
    )
    forward_per_atom = forward_output["mtt::U0"].block().values

    # Get per-atom predictions from _forward_from_batch
    (
        element_indices_nodes,
        element_indices_neighbors,
        edge_vectors,
        edge_distances,
        padding_mask,
        reverse_neighbor_index,
        cutoff_factors,
        system_indices,
        neighbor_atom_indices,
        sample_labels,
    ) = systems_to_batch(
        systems,
        model.requested_nl,
        model.atomic_types,
        model.species_to_species_index,
        model.cutoff_function,
        model.cutoff_width,
        model.num_neighbors_adaptive,
    )

    batch_output = model._forward_from_batch(
        element_indices_nodes,
        element_indices_neighbors,
        edge_vectors,
        edge_distances,
        padding_mask,
        reverse_neighbor_index,
        cutoff_factors,
    )
    # Get the first (and only) block key for the energy target
    energy_key = next(iter(model.output_shapes["mtt::U0"]))
    batch_per_atom = batch_output["mtt::U0"][energy_key]

    torch.testing.assert_close(forward_per_atom, batch_per_atom, atol=1e-6, rtol=1e-6)


def test_compiled_vs_eager_backward():
    """Compiled backward produces the same parameter gradients as eager.

    Energy-only (no forces/stress). Creates two identical PET models, runs
    forward + sum + backward on the same batch through both the eager
    ``_forward_from_batch`` path and the FX-compiled ``compile_pet_model``
    path, then compares every parameter gradient tensor.
    """
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader

    from metatrain.pet import PET
    from metatrain.utils.data import CollateFn, Dataset, DatasetInfo
    from metatrain.utils.data.readers import read_systems, read_targets
    from metatrain.utils.data.target_info import get_energy_target_info
    from metatrain.utils.neighbor_lists import (
        get_system_with_neighbor_lists,
        get_system_with_neighbor_lists_transform,
    )

    from ..modules.compile import compile_pet_model
    from ..modules.structures import systems_to_batch
    from ..modules.utilities import replace_silu_modules
    from . import DATASET_PATH, MODEL_HYPERS

    torch.manual_seed(42)

    targets = {
        "mtt::U0": get_energy_target_info(
            "mtt::U0", {"quantity": "energy", "unit": "eV"}
        )
    }
    dataset_info = DatasetInfo(
        length_unit="Angstrom", atomic_types=[1, 6, 7, 8], targets=targets
    )

    # Two identical models from the same seed
    torch.manual_seed(42)
    model_eager = PET(MODEL_HYPERS, dataset_info)
    torch.manual_seed(42)
    model_compiled = PET(MODEL_HYPERS, dataset_info)

    # compile_pet_model replaces nn.SiLU with DecomposedSiLU in-place;
    # apply the same replacement to the eager model for a fair comparison.
    replace_silu_modules(model_eager)

    # Load systems and add neighbor lists
    systems = read_systems(DATASET_PATH)[:5]
    systems = [s.to(torch.float32) for s in systems]
    for s in systems:
        get_system_with_neighbor_lists(s, model_eager.requested_neighbor_lists())

    # Shared batch input for both paths
    (
        elem_nodes,
        elem_neighbors,
        edge_vecs,
        edge_dists,
        pad_mask,
        rev_idx,
        cutoff_facs,
        sys_idx,
        nbr_atom_idx,
        _sample_labels,
    ) = systems_to_batch(
        systems,
        model_eager.requested_nl,
        model_eager.atomic_types,
        model_eager.species_to_species_index,
        model_eager.cutoff_function,
        model_eager.cutoff_width,
        model_eager.num_neighbors_adaptive,
    )

    # --- EAGER forward + backward ---
    model_eager.train()
    batch_output = model_eager._forward_from_batch(
        elem_nodes,
        elem_neighbors,
        edge_vecs,
        edge_dists,
        pad_mask,
        rev_idx,
        cutoff_facs,
    )
    energy_key = next(iter(model_eager.output_shapes["mtt::U0"]))
    per_atom_eager = batch_output["mtt::U0"][energy_key]
    loss_eager = per_atom_eager.sum()
    loss_eager.backward()

    grads_eager = {
        n: p.grad.clone()
        for n, p in model_eager.named_parameters()
        if p.grad is not None
    }

    # --- COMPILED forward + backward ---
    # Minimal DataLoader required by compile_pet_model for symbolic tracing
    conf = {
        "mtt::U0": {
            "quantity": "energy",
            "read_from": DATASET_PATH,
            "reader": "ase",
            "key": "U0",
            "unit": "eV",
            "type": "scalar",
            "per_atom": False,
            "num_subtargets": 1,
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    targets_data, _ = read_targets(OmegaConf.create(conf))
    raw_systems = read_systems(DATASET_PATH)[:5]
    targets_data["mtt::U0"] = targets_data["mtt::U0"][:5]
    dataset = Dataset.from_dict(
        {"system": raw_systems, "mtt::U0": targets_data["mtt::U0"]}
    )
    collate_fn = CollateFn(
        target_keys=["mtt::U0"],
        callables=[
            get_system_with_neighbor_lists_transform(
                model_compiled.requested_neighbor_lists()
            ),
        ],
    )
    dataloader = DataLoader(dataset, batch_size=5, shuffle=False, collate_fn=collate_fn)

    model_compiled.train()
    torch._dynamo.reset()
    compiled_fn, _, _ = compile_pet_model(
        model_compiled,
        dataloader,
        compute_forces=False,
        compute_stress=False,
    )

    energy_compiled, _, _, _ = compiled_fn(
        edge_vecs,
        elem_nodes,
        elem_neighbors,
        pad_mask,
        rev_idx,
        cutoff_facs,
        sys_idx,
        nbr_atom_idx,
        len(systems),
        *list(model_compiled.parameters()),
        *list(model_compiled.buffers()),
    )
    loss_compiled = energy_compiled.sum()
    loss_compiled.backward()

    grads_compiled = {
        n: p.grad.clone()
        for n, p in model_compiled.named_parameters()
        if p.grad is not None
    }

    # --- COMPARE ---
    # Forward: total energy must agree
    torch.testing.assert_close(
        per_atom_eager.sum(),
        energy_compiled.sum(),
        atol=1e-6,
        rtol=1e-6,
    )

    # Backward: every parameter gradient must agree
    assert set(grads_eager.keys()) == set(grads_compiled.keys()), (
        "Gradient keys differ between eager and compiled paths"
    )
    for name in grads_eager:
        torch.testing.assert_close(
            grads_eager[name],
            grads_compiled[name],
            atol=1e-5,
            rtol=1e-5,
            msg=f"Gradient mismatch: {name}",
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="compiled forces path needs SDPA double backward (CUDA only)",
)
def test_compiled_vs_eager_backward_with_forces():
    """Compiled backward with forces matches eager backward.

    The compiled path computes forces inside the FX graph via decomposed
    ``autograd.grad`` (create_graph=False). The eager path replicates the
    same force formula with ``create_graph=True`` so the outer backward
    can propagate through force contributions.

    Both paths should yield identical parameter gradients within tolerance.

    Requires CUDA: torch.compile's AOT autograd must differentiate through
    the force computation's backward pass (SDPA double backward), which is
    only implemented for CUDA.
    """
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader

    from metatrain.pet import PET
    from metatrain.utils.data import CollateFn, Dataset, DatasetInfo
    from metatrain.utils.data.readers import read_systems, read_targets
    from metatrain.utils.data.target_info import get_energy_target_info
    from metatrain.utils.neighbor_lists import (
        get_system_with_neighbor_lists,
        get_system_with_neighbor_lists_transform,
    )

    from ..modules.compile import compile_pet_model
    from ..modules.structures import systems_to_batch
    from ..modules.utilities import replace_silu_modules
    from . import DATASET_PATH, MODEL_HYPERS

    torch.manual_seed(42)

    targets = {
        "mtt::U0": get_energy_target_info(
            "mtt::U0", {"quantity": "energy", "unit": "eV"}
        )
    }
    dataset_info = DatasetInfo(
        length_unit="Angstrom", atomic_types=[1, 6, 7, 8], targets=targets
    )

    torch.manual_seed(42)
    model_eager = PET(MODEL_HYPERS, dataset_info)
    torch.manual_seed(42)
    model_compiled = PET(MODEL_HYPERS, dataset_info)

    replace_silu_modules(model_eager)

    systems = read_systems(DATASET_PATH)[:5]
    systems = [s.to(torch.float32) for s in systems]
    for s in systems:
        get_system_with_neighbor_lists(s, model_eager.requested_neighbor_lists())

    (
        elem_nodes,
        elem_neighbors,
        edge_vecs,
        edge_dists,
        pad_mask,
        rev_idx,
        cutoff_facs,
        sys_idx,
        nbr_atom_idx,
        _sample_labels,
    ) = systems_to_batch(
        systems,
        model_eager.requested_nl,
        model_eager.atomic_types,
        model_eager.species_to_species_index,
        model_eager.cutoff_function,
        model_eager.cutoff_width,
        model_eager.num_neighbors_adaptive,
    )

    n_structures = len(systems)
    n_atoms = edge_vecs.shape[0]

    # --- EAGER: forward + forces + backward ---
    # Replicate compile.py's force logic with create_graph=True
    model_eager.train()
    eager_edge_vecs = edge_vecs.clone().requires_grad_(True)
    eager_edge_dists = torch.sqrt((eager_edge_vecs**2).sum(-1) + 1e-15)
    batch_output = model_eager._forward_from_batch(
        elem_nodes,
        elem_neighbors,
        eager_edge_vecs,
        eager_edge_dists,
        pad_mask,
        rev_idx,
        cutoff_facs,
        use_manual_attention=True,  # needed for create_graph=True backward
    )
    energy_key = next(iter(model_eager.output_shapes["mtt::U0"]))
    per_atom_eager = batch_output["mtt::U0"][energy_key]

    # Aggregate per-atom to per-structure (same as compile.py)
    n_struct = n_structures + 1  # +1 for padding slot
    energy_eager = torch.zeros(
        n_struct,
        dtype=edge_vecs.dtype,
        device=edge_vecs.device,
    )
    energy_eager.scatter_add_(0, sys_idx, per_atom_eager.squeeze(-1))

    # Forces via autograd.grad with create_graph=True (eager double backward)
    (dE_dR_eager,) = torch.autograd.grad(
        energy_eager[:n_structures].sum(),
        eager_edge_vecs,
        create_graph=True,
    )
    dE_dR_eager = dE_dR_eager * pad_mask[:, :, None].float()
    grad_center = -dE_dR_eager.sum(dim=1)
    flat_dE = dE_dR_eager.reshape(-1, 3)
    flat_idx = nbr_atom_idx.reshape(-1, 1).expand(-1, 3).long()
    grad_neighbor = torch.zeros(
        n_atoms,
        3,
        dtype=edge_vecs.dtype,
        device=edge_vecs.device,
    )
    grad_neighbor.scatter_add_(0, flat_idx, flat_dE)
    forces_eager = grad_center + grad_neighbor

    loss_eager = energy_eager[:n_structures].sum() + forces_eager.sum()
    loss_eager.backward()

    grads_eager = {
        n: p.grad.clone()
        for n, p in model_eager.named_parameters()
        if p.grad is not None
    }

    # --- COMPILED: forward + forces + backward ---
    conf = {
        "mtt::U0": {
            "quantity": "energy",
            "read_from": DATASET_PATH,
            "reader": "ase",
            "key": "U0",
            "unit": "eV",
            "type": "scalar",
            "per_atom": False,
            "num_subtargets": 1,
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    targets_data, _ = read_targets(OmegaConf.create(conf))
    raw_systems = read_systems(DATASET_PATH)[:5]
    targets_data["mtt::U0"] = targets_data["mtt::U0"][:5]
    dataset = Dataset.from_dict(
        {"system": raw_systems, "mtt::U0": targets_data["mtt::U0"]}
    )
    collate_fn = CollateFn(
        target_keys=["mtt::U0"],
        callables=[
            get_system_with_neighbor_lists_transform(
                model_compiled.requested_neighbor_lists()
            ),
        ],
    )
    dataloader = DataLoader(dataset, batch_size=5, shuffle=False, collate_fn=collate_fn)

    model_compiled.train()
    torch._dynamo.reset()
    compiled_fn, _, _ = compile_pet_model(
        model_compiled,
        dataloader,
        compute_forces=True,
        compute_stress=False,
    )

    compiled_edge_vecs = edge_vecs.clone().requires_grad_(True)
    energy_compiled, forces_compiled, _, _ = compiled_fn(
        compiled_edge_vecs,
        elem_nodes,
        elem_neighbors,
        pad_mask,
        rev_idx,
        cutoff_facs,
        sys_idx,
        nbr_atom_idx,
        n_structures,
        *list(model_compiled.parameters()),
        *list(model_compiled.buffers()),
    )

    loss_compiled = energy_compiled.sum() + forces_compiled.sum()
    loss_compiled.backward()

    grads_compiled = {
        n: p.grad.clone()
        for n, p in model_compiled.named_parameters()
        if p.grad is not None
    }

    # --- COMPARE ---
    # Forward: energy and forces should match
    torch.testing.assert_close(
        energy_eager[:n_structures],
        energy_compiled,
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        forces_eager,
        forces_compiled,
        atol=1e-5,
        rtol=1e-5,
    )

    # Backward: parameter gradients should match (looser tolerance
    # because eager uses create_graph=True double backward while
    # compiled uses FX-decomposed single backward)
    assert set(grads_eager.keys()) == set(grads_compiled.keys()), (
        "Gradient keys differ between eager and compiled paths"
    )
    for name in grads_eager:
        torch.testing.assert_close(
            grads_eager[name],
            grads_compiled[name],
            atol=1e-4,
            rtol=1e-4,
            msg=f"Gradient mismatch: {name}",
        )


class TestTrainingCompile(TrainingTests, PETTests):
    """Run the standard training tests with compile=True.

    The full-graph FX compilation path traces the entire PET model
    (including force/stress computation) into a single FX graph and
    compiles it with ``torch.compile(dynamic=True, fullgraph=True)``.
    """

    @pytest.fixture
    def default_hypers(self):
        hypers = get_default_hypers(self.architecture)
        hypers["training"]["compile"] = True
        return hypers


class TestTrainingCompileAdaptive(TrainingTests, PETTests):
    """Run the standard training tests with compile=True and adaptive cutoffs.

    Adaptive cutoffs (num_neighbors_adaptive=16) cause the 2nd dimension
    of NEF tensors (max_edges_per_node) to vary per batch. This tests
    ``dynamic=True`` more aggressively than fixed cutoffs.
    """

    @pytest.fixture
    def default_hypers(self):
        hypers = get_default_hypers(self.architecture)
        hypers["training"]["compile"] = True
        hypers["model"]["num_neighbors_adaptive"] = 16
        return hypers
