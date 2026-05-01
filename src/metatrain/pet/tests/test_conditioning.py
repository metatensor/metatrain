"""Tests for charge and spin conditioning embeddings in PET."""

import copy

import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import ModelOutput, System

from metatrain.pet import PET
from metatrain.pet.modules.conditioning import SystemConditioningEmbedding
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import CollateFn, Dataset, DatasetInfo, unpack_batch
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.neighbor_lists import (
    get_system_with_neighbor_lists,
    get_system_with_neighbor_lists_transform,
)
from metatrain.utils.system_data import get_system_data_transform


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


def _make_system(model, charge=None, spin_multiplicity=None):
    """Create a simple 2-atom system with optional charge/spin_multiplicity."""
    system = System(
        types=torch.tensor([6, 1]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )
    if charge is not None:
        system.add_data("charge", _make_scalar_tmap(charge, "charge"))
    if spin_multiplicity is not None:
        system.add_data(
            "spin_multiplicity",
            _make_scalar_tmap(spin_multiplicity, "spin_multiplicity"),
        )
    return get_system_with_neighbor_lists(system, model.requested_neighbor_lists())


def test_conditioning_shapes():
    """SystemConditioningEmbedding produces [n_atoms, d_out]."""
    d_out = 16
    module = SystemConditioningEmbedding(
        d_out=d_out, max_charge=5, max_spin_multiplicity=5
    )

    charge = torch.tensor([0, 2])  # 2 systems
    spin_multiplicity = torch.tensor([1, 3])  # 2 systems
    system_indices = torch.tensor([0, 0, 0, 1, 1])  # 5 atoms total

    out = module(charge, spin_multiplicity, system_indices)
    assert out.shape == (5, d_out)
    assert out.dtype == torch.float32


def test_conditioning_different_d_node_d_pet():
    """Conditioning works when d_node != d_pet (the common case)."""
    hypers = _small_hypers(d_pet=8, d_node=16)
    model = PET(hypers, _dataset_info())
    model.eval()

    system = _make_system(model, charge=1, spin_multiplicity=2)
    outputs = {"energy": ModelOutput(per_atom=False)}
    with torch.no_grad():
        result = model([system], outputs)
    assert "energy" in result


def _train_steps(model, n_steps=20):
    """Do a few optimizer steps to break the zero-init of the conditioning gate."""
    torch.manual_seed(42)
    model.train()
    for _ in range(n_steps):
        system = _make_system(model, charge=2, spin_multiplicity=3)
        outputs = {"energy": ModelOutput(per_atom=False)}
        result = model([system], outputs)
        loss = result["energy"].block().values.sum()
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p -= 0.01 * p.grad
                    p.grad.zero_()
    model.eval()


def test_conditioning_changes_output():
    """Same structure with different charges should produce different predictions."""
    hypers = _small_hypers()
    model = PET(hypers, _dataset_info())
    _train_steps(model)

    system_neutral = _make_system(model, charge=0, spin_multiplicity=1)
    system_charged = _make_system(model, charge=2, spin_multiplicity=1)

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
    module = SystemConditioningEmbedding(d_out=8, max_charge=5, max_spin_multiplicity=5)

    charge = torch.tensor([1])
    spin_multiplicity = torch.tensor([2])
    system_indices = torch.tensor([0, 0])

    out = module(charge, spin_multiplicity, system_indices)
    loss = out.sum()
    loss.backward()

    assert module.charge_embedding.weight.grad is not None
    assert module.spin_multiplicity_embedding.weight.grad is not None
    assert module.project[0].weight.grad is not None


def test_conditioning_batch_independence():
    """Changing charge of one system in a batch should not affect others."""
    hypers = _small_hypers()
    model = PET(hypers, _dataset_info())
    _train_steps(model)

    system_a = _make_system(model, charge=0, spin_multiplicity=1)
    system_b_v1 = _make_system(model, charge=1, spin_multiplicity=1)
    system_b_v2 = _make_system(model, charge=3, spin_multiplicity=2)

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
    """Systems without explicit charge/spin_multiplicity should use defaults
    (charge=0, spin_multiplicity=1)."""
    hypers = _small_hypers()
    model = PET(hypers, _dataset_info())
    model.eval()

    # System with no charge/spin data (should default to charge=0, spin_multiplicity=1)
    system_default = _make_system(model)
    # System with explicit charge=0, spin_multiplicity=1
    system_explicit = _make_system(model, charge=0, spin_multiplicity=1)

    outputs = {"energy": ModelOutput(per_atom=False)}
    with torch.no_grad():
        result_default = model([system_default], outputs)
        result_explicit = model([system_explicit], outputs)

    e_default = result_default["energy"].block().values
    e_explicit = result_explicit["energy"].block().values
    torch.testing.assert_close(e_default, e_explicit)


def test_conditioning_out_of_range():
    """Charges or spins outside the supported range raise ValueError."""
    module = SystemConditioningEmbedding(d_out=8, max_charge=3, max_spin_multiplicity=4)

    # charge too positive
    with pytest.raises(ValueError, match=r"charge values must be in \[-3, 3\]"):
        module.validate(torch.tensor([5]), torch.tensor([1]))

    # charge too negative
    with pytest.raises(ValueError, match=r"charge values must be in \[-3, 3\]"):
        module.validate(torch.tensor([-4]), torch.tensor([1]))

    # spin too high
    with pytest.raises(
        ValueError, match=r"spin_multiplicity values must be in \[1, 4\]"
    ):
        module.validate(torch.tensor([0]), torch.tensor([5]))

    # spin too low (0 is invalid, minimum is 1)
    with pytest.raises(
        ValueError, match=r"spin_multiplicity values must be in \[1, 4\]"
    ):
        module.validate(torch.tensor([0]), torch.tensor([0]))


def _make_raw_system() -> System:
    """2-atom system without neighbor lists (as stored in a Dataset).

    Uses float64 to match what read_systems() produces; batch_to() later
    converts to the model's float32.
    """
    return System(
        types=torch.tensor([6, 1]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.1]], dtype=torch.float64),
        cell=torch.zeros(3, 3, dtype=torch.float64),
        pbc=torch.tensor([False, False, False]),
    )


def _make_extra_tmap(value: float, prop_name: str) -> TensorMap:
    return TensorMap(
        keys=Labels.single(),
        blocks=[
            TensorBlock(
                values=torch.tensor([[value]], dtype=torch.float64),
                samples=Labels("system", torch.tensor([[0]])),
                components=[],
                properties=Labels(prop_name, torch.tensor([[0]])),
            )
        ],
    )


def test_eval_routes_extra_data_to_conditioning():
    """Regression: extra_data charge/spin must reach system conditioning during eval.

    Simulates the eval data pipeline (Dataset + CollateFn with
    get_system_data_transform) and verifies that a non-default charge produces
    different model predictions than the default (charge=0).

    This would silently fail if extra_data_keys were filtered to empty (e.g.
    by incorrectly gating on model.requested_inputs() which returns {} for
    exported models), causing all systems to fall back to
    charge=0 / spin_multiplicity=1.
    """
    hypers = _small_hypers()
    model = PET(hypers, _dataset_info())
    _train_steps(model)
    model.eval()

    extra_data_keys = ["charge", "spin_multiplicity"]

    dataset_neutral = Dataset.from_dict(
        {
            "system": [_make_raw_system()],
            "charge": [_make_extra_tmap(0.0, "charge")],
            "spin_multiplicity": [_make_extra_tmap(1.0, "spin_multiplicity")],
        }
    )
    dataset_charged = Dataset.from_dict(
        {
            "system": [_make_raw_system()],
            "charge": [_make_extra_tmap(2.0, "charge")],  # non-default
            "spin_multiplicity": [_make_extra_tmap(1.0, "spin_multiplicity")],
        }
    )

    callables = [
        get_system_with_neighbor_lists_transform(model.requested_neighbor_lists()),
        get_system_data_transform(extra_data_keys),
    ]
    collate_fn = CollateFn(["energy"], callables=callables)

    systems_neutral, _, _ = unpack_batch(collate_fn([dataset_neutral[0]]))
    systems_charged, _, _ = unpack_batch(collate_fn([dataset_charged[0]]))

    # Verify charge was actually attached to systems — if the transform is
    # skipped, known_data() will be empty and the assertion fails here.
    assert "charge" in systems_neutral[0].known_data(), (
        "charge not attached to neutral system — "
        "get_system_data_transform was not applied"
    )
    assert "charge" in systems_charged[0].known_data(), (
        "charge not attached to charged system — "
        "get_system_data_transform was not applied"
    )

    charge_neutral = systems_neutral[0].get_data("charge").block().values.item()
    charge_charged = systems_charged[0].get_data("charge").block().values.item()
    assert charge_neutral == 0.0
    assert charge_charged == 2.0

    # Convert to float32 (model dtype), mirroring batch_to() in the eval loop
    systems_neutral = [s.to(dtype=torch.float32) for s in systems_neutral]
    systems_charged = [s.to(dtype=torch.float32) for s in systems_charged]

    outputs = {"energy": ModelOutput(per_atom=False)}
    with torch.no_grad():
        e_neutral = model(systems_neutral, outputs)["energy"].block().values
        e_charged = model(systems_charged, outputs)["energy"].block().values

    assert not torch.allclose(e_neutral, e_charged), (
        "charge=0 and charge=2 produce identical energies — "
        "conditioning is not being applied (extra_data not routed to systems)"
    )


@pytest.mark.filterwarnings(
    "ignore:the 'features' output name is deprecated:UserWarning",
    "ignore:`per_atom` is deprecated:DeprecationWarning",
    "ignore:ModelOutput.quantity is deprecated:UserWarning",
    "ignore:Found metatomic.torch v.*vesin.metatomic was only tested:UserWarning",
    "ignore:calling Model.requested_inputs.use_new_names=False. is deprecated"
    ":UserWarning",
)
def test_export_with_conditioning_preserves_validate(tmp_path):
    """Exporting (TorchScript) a PET model with system_conditioning must keep the
    in-forward ``validate(...)`` call working: ``mtt export`` succeeds, valid
    inputs run through, and an out-of-range value raises a clear error from the
    scripted module (driven via ``MetatomicCalculator`` for end-to-end coverage).
    """
    import ase
    from metatomic_ase import MetatomicCalculator

    hypers = _small_hypers(max_charge=3, max_spin_multiplicity=4)
    model = PET(hypers, _dataset_info())
    model.eval()

    path = str(tmp_path / "pet_conditioning.pt")
    # If validate() did not survive scripting, this line raises during export.
    model.export().save(path)

    calculator = MetatomicCalculator(path)

    # Valid inputs (spin_multiplicity within [1, max_spin_multiplicity]) work.
    atoms = ase.Atoms("CH", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    atoms.info["charge"] = 2
    atoms.info["spin"] = 3
    atoms.calc = calculator
    _ = atoms.get_potential_energy()

    # Out-of-range spin_multiplicity must be rejected by the scripted validate().
    atoms_bad = ase.Atoms("CH", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    atoms_bad.info["charge"] = 0
    atoms_bad.info["spin"] = 99
    atoms_bad.calc = MetatomicCalculator(path)
    with pytest.raises(Exception, match="spin_multiplicity"):
        atoms_bad.get_potential_energy()
