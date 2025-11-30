import pytest
import torch
from metatomic.torch import ModelOutput, System

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
    outputs = {"energy": ModelOutput(per_atom=False)}
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


def test_consistency():
    """Tests that the two implementations of attention are consistent."""

    num_centers = 100
    num_neighbors_per_center = 50
    hidden_size = 128
    num_heads = 4

    attention = AttentionBlock(hidden_size, num_heads)

    inputs = torch.randn(num_centers, num_neighbors_per_center, hidden_size)
    radial_mask = torch.rand(
        num_centers, num_neighbors_per_center, num_neighbors_per_center
    )

    attention_output_torch = attention(inputs, radial_mask, use_manual_attention=False)
    attention_output_manual = attention(inputs, radial_mask, use_manual_attention=True)

    assert torch.allclose(attention_output_torch, attention_output_manual, atol=1e-6)


@pytest.mark.parametrize("per_atom", [True, False])
def test_nc_stress(per_atom):
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
                    "per_atom": per_atom,
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
    outputs = {"non_conservative_stress": ModelOutput(per_atom=per_atom)}
    stress = model([system], outputs)["non_conservative_stress"].block().values
    assert torch.allclose(stress, stress.transpose(1, 2))
