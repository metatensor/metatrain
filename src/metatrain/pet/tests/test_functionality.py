import copy

import ase.build
import ase.units
import pytest
import torch
from ase.md import VelocityVerlet
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import ModelOutput, System
from metatomic_ase import MetatomicCalculator

from metatrain.composition import Trainer as CompositionTrainer
from metatrain.pet import PET
from metatrain.pet.modules.transformer import AttentionBlock
from metatrain.utils.data import Dataset, DatasetInfo
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_attention_above_cuda_grid_limit():
    """Above 65535 nodes, SDPA's CUDA backward overflows the grid limit and crashes;
    AttentionBlock must fall back to a for-loop over batch chunks of SDPA, warning
    that this may reduce performance. Skipped without a GPU."""

    device = "cuda"
    # Just over the grid limit; tiny other dims keep it cheap.
    batch = 65536
    seq_length = 1
    num_heads = 1
    head_dim = 4
    hidden_size = num_heads * head_dim

    # Where the flash/mem-efficient kernel is used, the raw SDPA backward overflows
    # the grid limit.
    queries = torch.randn(
        batch, num_heads, seq_length, head_dim, device=device, requires_grad=True
    )
    keys = torch.randn_like(queries)
    values = torch.randn_like(queries)
    out = torch.nn.functional.scaled_dot_product_attention(queries, keys, values)
    try:
        out.sum().backward()
    except RuntimeError as error:
        assert "65535" in str(error)

    # AttentionBlock auto-falls back to a chunked for-loop over SDPA, warning about
    # the reduced performance, and succeeds.
    attention = AttentionBlock(hidden_size, num_heads, temperature=1.0).to(device)
    inputs = torch.randn(
        batch, seq_length, hidden_size, device=device, requires_grad=True
    )
    cutoff_factors = torch.rand(batch, seq_length, seq_length, device=device)
    with pytest.warns(UserWarning, match="CUDA grid dimension"):
        output = attention(inputs, cutoff_factors, use_manual_attention=False)
    output.sum().backward()
    assert inputs.grad is not None
    assert inputs.grad.shape == inputs.shape


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


def test_composition_contribution_in_eval_atomic_basis():
    """Tests that the composition contribution is present in the eval-mode
    output for an atomic-basis target.

    In eval mode, PET sparsifies its own atomic-basis predictions and its
    additive models sparsify their contributions; the two are then summed
    block by block. This test guards against the contribution being silently
    dropped by the key-matched sum (which skips blocks whose keys don't
    match).
    """
    irreps = {
        1: [{"o3_lambda": 0, "o3_sigma": 1}],
        8: [
            {"o3_lambda": 0, "o3_sigma": 1},
            {"o3_lambda": 1, "o3_sigma": 1},
        ],
    }
    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 8],
        targets={
            "mtt::basis": get_generic_target_info(
                "mtt::basis",
                {
                    "quantity": "",
                    "unit": "",
                    "type": {"spherical": {"irreps": irreps}},
                    "num_subtargets": 1,
                    "sample_kind": "atom",
                },
            )
        },
    )

    model = PET(MODEL_HYPERS, dataset_info)
    # Identical PET weights, but its composition model stays unfitted (zero
    # weights): the difference between the two eval outputs must be exactly
    # the composition contribution.
    model_no_composition = copy.deepcopy(model)

    # Fit the composition model of ``model`` on two synthetic structures:
    # - O atom, with an invariant of 1.0
    # - H2O molecule, with invariants of 1.0, 1.5, 2.0
    # The expected composition weights are 1.25 for H and 1.5 for O.
    systems = [
        System(
            positions=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64),
            types=torch.tensor([8]),
            cell=torch.eye(3, dtype=torch.float64),
            pbc=torch.tensor([True, True, True]),
        ),
        System(
            positions=torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=torch.float64,
            ),
            types=torch.tensor([1, 1, 8]),
            cell=torch.eye(3, dtype=torch.float64),
            pbc=torch.tensor([True, True, True]),
        ),
    ]
    targets = [
        TensorMap(
            keys=Labels(
                names=["o3_lambda", "o3_sigma", "atom_type"],
                values=torch.tensor([[0, 1, 8]]),
            ),
            blocks=[
                TensorBlock(
                    values=torch.tensor([[1.0]], dtype=torch.float64).reshape(-1, 1, 1),
                    samples=Labels(
                        names=["system", "atom"], values=torch.tensor([[0, 0]])
                    ),
                    components=[Labels(names=["o3_mu"], values=torch.tensor([[0]]))],
                    properties=Labels(names=["n"], values=torch.tensor([[0]])),
                ),
            ],
        ),
        TensorMap(
            keys=Labels(
                names=["o3_lambda", "o3_sigma", "atom_type"],
                values=torch.tensor([[0, 1, 1], [0, 1, 8]]),
            ),
            blocks=[
                TensorBlock(
                    values=torch.tensor([[1.0], [1.5]], dtype=torch.float64).reshape(
                        -1, 1, 1
                    ),
                    samples=Labels(
                        names=["system", "atom"],
                        values=torch.tensor([[1, 0], [1, 1]]),
                    ),
                    components=[Labels(names=["o3_mu"], values=torch.tensor([[0]]))],
                    properties=Labels(names=["n"], values=torch.tensor([[0]])),
                ),
                TensorBlock(
                    values=torch.tensor([[2.0]], dtype=torch.float64).reshape(-1, 1, 1),
                    samples=Labels(
                        names=["system", "atom"], values=torch.tensor([[1, 2]])
                    ),
                    components=[Labels(names=["o3_mu"], values=torch.tensor([[0]]))],
                    properties=Labels(names=["n"], values=torch.tensor([[0]])),
                ),
            ],
        ),
    ]
    system_indices = [
        TensorMap(
            keys=Labels(names=["_"], values=torch.tensor([[0]])),
            blocks=[
                TensorBlock(
                    values=torch.tensor([[i]], dtype=torch.float64),
                    samples=Labels(names=["system"], values=torch.tensor([[i]])),
                    components=[],
                    properties=Labels(names=["_"], values=torch.tensor([[0]])),
                )
            ],
        )
        for i in range(len(systems))
    ]
    dataset = Dataset.from_dict(
        {
            "system": systems,
            "mtt::basis": targets,
            "mtt::aux::system_index": system_indices,
        }
    )
    composition_trainer = CompositionTrainer(
        hypers={"atomic_baseline": {}, "batch_size": 1}
    )
    composition_trainer.train(
        model=model.additive_models[0],
        dtype=torch.float64,
        devices=[torch.device("cpu")],
        train_datasets=[dataset],
        val_datasets=[],
        checkpoint_dir="",
    )

    model.eval()
    model_no_composition.eval()

    system = System(
        types=torch.tensor([1, 8]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )
    system = get_system_with_neighbor_lists(system, model.requested_neighbor_lists())
    outputs = {"mtt::basis": ModelOutput(sample_kind="atom")}
    out = model([system], outputs)["mtt::basis"]
    out_no_composition = model_no_composition([system], outputs)["mtt::basis"]

    # Both outputs are in the target's native sparse layout
    assert out.keys.names == ["o3_lambda", "o3_sigma", "atom_type"]
    assert out.keys == out_no_composition.keys

    # The composition contribution is present in the invariant blocks
    h_key = {"o3_lambda": 0, "o3_sigma": 1, "atom_type": 1}
    o_key = {"o3_lambda": 0, "o3_sigma": 1, "atom_type": 8}
    torch.testing.assert_close(
        out.block(h_key).values - out_no_composition.block(h_key).values,
        torch.full_like(out.block(h_key).values, 1.25),
    )
    torch.testing.assert_close(
        out.block(o_key).values - out_no_composition.block(o_key).values,
        torch.full_like(out.block(o_key).values, 1.5),
    )
    # ...and the (unfitted) l=1 block is unaffected
    o_l1_key = {"o3_lambda": 1, "o3_sigma": 1, "atom_type": 8}
    torch.testing.assert_close(
        out.block(o_l1_key).values, out_no_composition.block(o_l1_key).values
    )
