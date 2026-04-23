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
    outputs = {"energy": ModelOutput(per_atom=False)}
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
    outputs = {"energy": ModelOutput(per_atom=False)}
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
    outputs = {"energy": ModelOutput(per_atom=False)}
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


def test_diagnostic_feature_hooks_capture_node_and_edge_features() -> None:
    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "energy": get_energy_target_info(
                "energy", {"quantity": "energy", "unit": "eV"}
            )
        },
    )

    model = PET(MODEL_HYPERS, dataset_info).eval()
    system = System(
        types=torch.tensor([6, 6]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )
    system = get_system_with_neighbor_lists(system, model.requested_neighbor_lists())
    outputs = {
        "energy": ModelOutput(per_atom=False),
        "mtt::features::gnn_layers.0_node": ModelOutput(per_atom=True),
        "mtt::features::gnn_layers.0_edge": ModelOutput(per_atom=True),
        "mtt::aux::energy_last_layer_features": ModelOutput(per_atom=True),
    }

    result = model([system], outputs)

    node_features = result["mtt::features::gnn_layers.0_node"].block()
    edge_features = result["mtt::features::gnn_layers.0_edge"].block()
    last_layer_features = result["mtt::aux::energy_last_layer_features"].block()

    assert node_features.samples.names == ["system", "atom"]
    assert node_features.values.shape[0] == 2
    assert edge_features.samples.names == [
        "system",
        "first_atom",
        "second_atom",
        "cell_shift_a",
        "cell_shift_b",
        "cell_shift_c",
    ]
    assert edge_features.values.shape[0] > 0
    assert last_layer_features.samples.names == ["system", "atom"]
    assert last_layer_features.values.shape[0] == 2


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


def test_volume_normalized_spherical_target_scales_with_inverse_volume():
    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "quadrupole": get_generic_target_info(
                "quadrupole",
                {
                    "quantity": "quadrupole",
                    "unit": "",
                    "type": {"spherical": {"irreps": [{"o3_lambda": 2, "o3_sigma": 1}]}},
                    "num_subtargets": 1,
                    "per_atom": False,
                },
            )
        },
    )

    raw_model_hypers = dict(MODEL_HYPERS)
    normalized_model_hypers = dict(MODEL_HYPERS)
    normalized_model_hypers["volume_normalized_targets"] = ["quadrupole"]

    torch.manual_seed(17)
    raw_model = PET(raw_model_hypers, dataset_info)
    torch.manual_seed(17)
    normalized_model = PET(normalized_model_hypers, dataset_info)

    system = System(
        types=torch.tensor([6]),
        positions=torch.tensor([[0.0, 0.0, 1.0]]),
        cell=torch.eye(3) * 4.0,
        pbc=torch.tensor([True, True, True]),
    )
    system_raw = get_system_with_neighbor_lists(
        system, raw_model.requested_neighbor_lists()
    )
    system_normalized = get_system_with_neighbor_lists(
        system, normalized_model.requested_neighbor_lists()
    )
    outputs = {"quadrupole": ModelOutput(per_atom=False)}
    raw_prediction = raw_model([system_raw], outputs)["quadrupole"].block().values
    normalized_prediction = normalized_model([system_normalized], outputs)[
        "quadrupole"
    ].block().values

    assert torch.allclose(
        raw_prediction / 64.0, normalized_prediction, atol=1e-6, rtol=1e-6
    )


def test_unknown_volume_normalized_target_is_rejected():
    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "energy": get_energy_target_info(
                "energy", {"quantity": "energy", "unit": "eV"}
            )
        },
    )
    model_hypers = dict(MODEL_HYPERS)
    model_hypers["volume_normalized_targets"] = ["ghost"]

    with pytest.raises(ValueError, match="Unknown volume-normalized target names"):
        PET(model_hypers, dataset_info)


def _mixed_shared_head_dataset_info() -> DatasetInfo:
    return DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "scalar": get_generic_target_info(
                "scalar",
                {
                    "quantity": "stress_l0",
                    "unit": "",
                    "type": "scalar",
                    "per_atom": False,
                    "num_subtargets": 1,
                },
            ),
            "quadrupole": get_generic_target_info(
                "quadrupole",
                {
                    "quantity": "stress_l2",
                    "unit": "",
                    "type": {
                        "spherical": {
                            "irreps": [{"o3_lambda": 2, "o3_sigma": 1}]
                        }
                    },
                    "per_atom": False,
                    "num_subtargets": 1,
                },
            ),
        },
    )


def test_shared_head_groups_share_heads_across_scalar_and_spherical_targets() -> None:
    model_hypers = dict(MODEL_HYPERS)
    model_hypers["shared_head_groups"] = {
        "stress_head": ["scalar", "quadrupole[2,1]"]
    }
    model = PET(model_hypers, _mixed_shared_head_dataset_info())

    assert model.target_head_keys["scalar"] == ["shared__stress_head"]
    assert model.target_head_keys["quadrupole"] == ["shared__stress_head"]
    assert "shared__stress_head" in model.node_heads
    assert "shared__stress_head" in model.edge_heads
    assert "scalar" not in model.node_heads
    assert "scalar" not in model.edge_heads
    assert model.node_last_layers["scalar"] is not model.node_last_layers["quadrupole"]
    assert model.edge_last_layers["scalar"] is not model.edge_last_layers["quadrupole"]


def test_shared_head_groups_reject_scalar_irrep_selector() -> None:
    model_hypers = dict(MODEL_HYPERS)
    model_hypers["shared_head_groups"] = {"bad": ["scalar[0,1]"]}

    with pytest.raises(
        ValueError, match="Scalar selectors cannot include an irrep suffix"
    ):
        PET(model_hypers, _mixed_shared_head_dataset_info())


def test_shared_head_groups_reject_missing_irrep_for_spherical_target() -> None:
    model_hypers = dict(MODEL_HYPERS)
    model_hypers["shared_head_groups"] = {"bad": ["quadrupole"]}

    with pytest.raises(
        ValueError, match="must include an explicit irrep suffix"
    ):
        PET(model_hypers, _mixed_shared_head_dataset_info())


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
