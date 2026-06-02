import ase.build
import ase.units
import pytest
import torch
from ase.md import VelocityVerlet
from metatomic.torch import ModelOutput, System
from metatomic_ase import MetatomicCalculator

from metatrain.pet import PET
from metatrain.pet.modules.transformer import AttentionBlock
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import (
    get_energy_target_info,
    get_generic_target_info,
)
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from . import MODEL_HYPERS


def test_pet_padding():
    """Tests that the model predicts the same energy independently of the
    padding size."""

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "energy": get_energy_target_info(
                "energy", {"quantity": "energy", "unit": "eV"}
            )
        },
    )

    model = PET(MODEL_HYPERS, dataset_info)

    system = System(
        types=torch.tensor([6, 6]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )
    system = get_system_with_neighbor_lists(system, model.requested_neighbor_lists())
    outputs = {"energy": ModelOutput(sample_kind="system")}
    lone_output = model([system], outputs)

    system_2 = System(
        types=torch.tensor([6, 6, 6, 6, 6, 6, 6]),
        positions=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 3.0],
                [0.0, 0.0, 4.0],
                [0.0, 0.0, 5.0],
                [0.0, 0.0, 6.0],
            ]
        ),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )
    system_2 = get_system_with_neighbor_lists(
        system_2, model.requested_neighbor_lists()
    )
    padded_output = model([system, system_2], outputs)

    lone_energy = lone_output["energy"].block().values.squeeze(-1)[0]
    padded_energy = padded_output["energy"].block().values.squeeze(-1)[0]

    assert torch.allclose(lone_energy, padded_energy, atol=1e-6, rtol=1e-6)


def test_empty_system():
    """Tests that the model can handle an empty system."""

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "energy": get_energy_target_info(
                "energy", {"quantity": "energy", "unit": "eV"}
            )
        },
    )

    model = PET(MODEL_HYPERS, dataset_info)

    system = System(
        types=torch.tensor([], dtype=torch.long),
        positions=torch.empty((0, 3)),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )
    system = get_system_with_neighbor_lists(system, model.requested_neighbor_lists())
    outputs = {"energy": ModelOutput(sample_kind="system")}
    energy = model([system], outputs)["energy"].block().values.squeeze(-1)
    assert torch.numel(energy) == 0


def test_isolated_atoms():
    """Tests that the model can predict energies for isolated atoms."""

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "energy": get_energy_target_info(
                "energy", {"quantity": "energy", "unit": "eV"}
            )
        },
    )

    model = PET(MODEL_HYPERS, dataset_info)

    system = System(
        types=torch.tensor([6]),
        positions=torch.tensor([[0.0, 0.0, 0.0]]),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )
    system = get_system_with_neighbor_lists(system, model.requested_neighbor_lists())
    outputs = {"energy": ModelOutput(sample_kind="system")}
    energy = model([system], outputs)["energy"].block().values.squeeze(-1)[0]

    assert torch.isfinite(energy)


def test_dissociated_atoms():
    """Tests that the model can predict energies for dissociated atoms."""

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "energy": get_energy_target_info(
                "energy", {"quantity": "energy", "unit": "eV"}
            )
        },
    )

    model = PET(MODEL_HYPERS, dataset_info)

    system = System(
        types=torch.tensor([6, 6]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 100.0]]),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )
    system = get_system_with_neighbor_lists(system, model.requested_neighbor_lists())
    outputs = {"energy": ModelOutput(sample_kind="system")}
    energy = model([system], outputs)["energy"].block().values.squeeze(-1)[0]

    assert torch.isfinite(energy)


def test_consistency():
    """Tests that the two implementations of attention are consistent."""

    num_centers = 100
    num_neighbors_per_center = 50
    hidden_size = 128
    num_heads = 4
    temperature = 2.0

    attention = AttentionBlock(hidden_size, num_heads, temperature)

    inputs = torch.randn(num_centers, num_neighbors_per_center, hidden_size)
    radial_mask = torch.rand(
        num_centers, num_neighbors_per_center, num_neighbors_per_center
    )

    attention_output_torch = attention(inputs, radial_mask, use_manual_attention=False)
    attention_output_manual = attention(inputs, radial_mask, use_manual_attention=True)

    assert torch.allclose(attention_output_torch, attention_output_manual, atol=1e-6)


@pytest.mark.parametrize("sample_kind", ["atom", "system"])
def test_nc_stress(sample_kind):
    """Tests that the model can predict a symmetric rank-2 tensor as the NC stress."""
    # (note that no composition energies are supplied or calculated here)

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "non_conservative_stress": get_generic_target_info(
                "non_conservative_stress",
                {
                    "quantity": "stress",
                    "unit": "",
                    "type": {"cartesian": {"rank": 2}},
                    "num_subtargets": 100,
                    "sample_kind": sample_kind,
                },
            )
        },
    )

    model = PET(MODEL_HYPERS, dataset_info)

    system = System(
        types=torch.tensor([6]),
        positions=torch.tensor([[0.0, 0.0, 1.0]]),
        cell=torch.eye(3),
        pbc=torch.tensor([True, True, True]),
    )
    system = get_system_with_neighbor_lists(system, model.requested_neighbor_lists())
    outputs = {"non_conservative_stress": ModelOutput(sample_kind=sample_kind)}
    stress = model([system], outputs)["non_conservative_stress"].block().values
    assert torch.allclose(stress, stress.transpose(1, 2))


def test_isolated_atom(monkeypatch, tmp_path):
    """Test that a short MD run completes without errors on an isolated atom."""
    monkeypatch.chdir(tmp_path)

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "energy": get_energy_target_info(
                "energy", {"quantity": "energy", "unit": "eV"}
            )
        },
    )
    model = PET(MODEL_HYPERS, dataset_info)
    model.export().save("pet.pt")

    atoms = ase.Atoms("O", positions=[[0, 0, 0]])

    time_step = 64
    device = "cuda" if torch.cuda.is_available() else "cpu"
    calculator = MetatomicCalculator("pet.pt", device=device)
    atoms.calc = calculator

    dyn = VelocityVerlet(atoms=atoms, timestep=time_step * ase.units.fs)
    dyn.run(3)


def test_slab_plus_isolated_atom(monkeypatch, tmp_path):
    """
    Test that a short MD run completes without errors on a slab plus an isolated atom.
    """
    monkeypatch.chdir(tmp_path)

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[8, 13, 14],
        targets={
            "energy": get_energy_target_info(
                "energy", {"quantity": "energy", "unit": "eV"}
            )
        },
    )
    model = PET(MODEL_HYPERS, dataset_info)
    model.export().save("pet.pt")

    # Create a slab and an isolated atom
    slab = ase.build.fcc111("Al", size=(2, 2, 3), vacuum=10)
    isolated_atom = ase.Atoms("O", positions=[[0, 0, 24]])
    atoms = slab + isolated_atom

    time_step = 64
    device = "cuda" if torch.cuda.is_available() else "cpu"
    calculator = MetatomicCalculator("pet.pt", device=device)
    atoms.calc = calculator

    dyn = VelocityVerlet(atoms=atoms, timestep=time_step * ase.units.fs)
    dyn.run(3)
