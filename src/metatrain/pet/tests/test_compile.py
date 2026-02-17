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
