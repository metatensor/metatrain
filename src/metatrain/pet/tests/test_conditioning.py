"""Tests for charge and spin conditioning embeddings in PET."""

import copy

import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import ModelOutput, System

from metatrain.pet import PET
from metatrain.pet.modules.conditioning import SystemConditioningEmbedding
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists


def _small_hypers(system_conditioning=True, **kwargs):
    """Return minimal PET hypers with system conditioning."""
    hypers = copy.deepcopy(get_default_hypers("pet")["model"])
    hypers["d_pet"] = 8
    hypers["d_head"] = 8
    hypers["d_node"] = 8
    hypers["d_feedforward"] = 8
    hypers["num_heads"] = 1
    hypers["num_attention_layers"] = 1
    hypers["num_gnn_layers"] = 1
    hypers["system_conditioning"] = system_conditioning
    hypers.update(kwargs)
    return hypers


def _dataset_info():
    return DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6],
        targets={
            "energy": get_energy_target_info(
                "energy", {"quantity": "energy", "unit": "eV"}
            )
        },
    )


def _make_scalar_tmap(value: int, property_name: str = "value") -> TensorMap:
    """Create a scalar TensorMap with a single float value (System requires float)."""
    return TensorMap(
        keys=Labels.single(),
        blocks=[
            TensorBlock(
                values=torch.tensor([[float(value)]]),
                samples=Labels("system", torch.tensor([[0]])),
                components=[],
                properties=Labels(property_name, torch.tensor([[0]])),
            )
        ],
    )


def _make_system(model, charge=None, spin=None):
    """Create a simple 2-atom system with optional charge/spin."""
    system = System(
        types=torch.tensor([6, 1]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )
    if charge is not None:
        system.add_data("mtt::charge", _make_scalar_tmap(charge, "charge"))
    if spin is not None:
        system.add_data("mtt::spin", _make_scalar_tmap(spin, "spin"))
    return get_system_with_neighbor_lists(system, model.requested_neighbor_lists())


def test_conditioning_shapes():
    """SystemConditioningEmbedding produces [n_atoms, d_out]."""
    d_out = 16
    module = SystemConditioningEmbedding(d_out=d_out, max_charge=5, max_spin=5)

    charge = torch.tensor([0, 2])  # 2 systems
    spin = torch.tensor([1, 3])  # 2 systems
    system_indices = torch.tensor([0, 0, 0, 1, 1])  # 5 atoms total

    out = module(charge, spin, system_indices)
    assert out.shape == (5, d_out)
    assert out.dtype == torch.float32


def test_conditioning_changes_output():
    """Same structure with different charges should produce different predictions."""
    hypers = _small_hypers()
    model = PET(hypers, _dataset_info())
    model.eval()

    system_neutral = _make_system(model, charge=0, spin=1)
    system_charged = _make_system(model, charge=2, spin=1)

    outputs = {"energy": ModelOutput(per_atom=False)}
    with torch.no_grad():
        result_neutral = model([system_neutral], outputs)
        result_charged = model([system_charged], outputs)

    e_neutral = result_neutral["energy"].block().values
    e_charged = result_charged["energy"].block().values
    assert not torch.allclose(e_neutral, e_charged), (
        "Different charges should produce different energies"
    )


def test_conditioning_disabled_unchanged():
    """With system_conditioning=False, no conditioning module should exist."""
    hypers_off = _small_hypers(system_conditioning=False)
    model = PET(hypers_off, _dataset_info())

    assert model.system_conditioning is None

    # Model should still run fine
    model.eval()
    system = _make_system(model)
    outputs = {"energy": ModelOutput(per_atom=False)}
    with torch.no_grad():
        result = model([system], outputs)
    assert "energy" in result


def test_conditioning_gradients_flow():
    """Gradients should flow through the conditioning embeddings."""
    module = SystemConditioningEmbedding(d_out=8, max_charge=5, max_spin=5)

    charge = torch.tensor([1])
    spin = torch.tensor([2])
    system_indices = torch.tensor([0, 0])

    out = module(charge, spin, system_indices)
    loss = out.sum()
    loss.backward()

    assert module.charge_embedding.weight.grad is not None
    assert module.spin_embedding.weight.grad is not None
    assert module.project[0].weight.grad is not None


def test_conditioning_batch_independence():
    """Changing charge of one system in a batch should not affect others."""
    hypers = _small_hypers()
    model = PET(hypers, _dataset_info())
    model.eval()

    system_a = _make_system(model, charge=0, spin=1)
    system_b_v1 = _make_system(model, charge=1, spin=1)
    system_b_v2 = _make_system(model, charge=3, spin=2)

    outputs = {"energy": ModelOutput(per_atom=False)}
    with torch.no_grad():
        result_v1 = model([system_a, system_b_v1], outputs)
        result_v2 = model([system_a, system_b_v2], outputs)

    # Energy of system_a should be the same in both batches
    e_a_v1 = result_v1["energy"].block().values[0]
    e_a_v2 = result_v2["energy"].block().values[0]
    torch.testing.assert_close(e_a_v1, e_a_v2)

    # Energy of system_b should differ between batches
    e_b_v1 = result_v1["energy"].block().values[1]
    e_b_v2 = result_v2["energy"].block().values[1]
    assert not torch.allclose(e_b_v1, e_b_v2)


def test_conditioning_default_values():
    """Systems without explicit charge/spin should use defaults (charge=0, spin=1)."""
    hypers = _small_hypers()
    model = PET(hypers, _dataset_info())
    model.eval()

    # System with no charge/spin data (should default to charge=0, spin=1)
    system_default = _make_system(model)
    # System with explicit charge=0, spin=1
    system_explicit = _make_system(model, charge=0, spin=1)

    outputs = {"energy": ModelOutput(per_atom=False)}
    with torch.no_grad():
        result_default = model([system_default], outputs)
        result_explicit = model([system_explicit], outputs)

    e_default = result_default["energy"].block().values
    e_explicit = result_explicit["energy"].block().values
    torch.testing.assert_close(e_default, e_explicit)


def test_conditioning_out_of_range():
    """Charges or spins outside the supported range raise ValueError."""
    module = SystemConditioningEmbedding(d_out=8, max_charge=3, max_spin=4)
    system_indices = torch.tensor([0])

    # charge too positive
    with pytest.raises(ValueError, match=r"charge values must be in \[-3, 3\]"):
        module(torch.tensor([5]), torch.tensor([1]), system_indices)

    # charge too negative
    with pytest.raises(ValueError, match=r"charge values must be in \[-3, 3\]"):
        module(torch.tensor([-4]), torch.tensor([1]), system_indices)

    # spin too high
    with pytest.raises(
        ValueError, match=r"spin multiplicity values must be in \[1, 4\]"
    ):
        module(torch.tensor([0]), torch.tensor([5]), system_indices)

    # spin too low (0 is invalid, minimum is 1)
    with pytest.raises(
        ValueError, match=r"spin multiplicity values must be in \[1, 4\]"
    ):
        module(torch.tensor([0]), torch.tensor([0]), system_indices)
