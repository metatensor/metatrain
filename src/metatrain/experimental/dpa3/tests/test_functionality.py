import metatensor.torch as mts
import torch
from metatomic.torch import ModelOutput, System
from omegaconf import OmegaConf

from metatrain.experimental.dpa3 import DPA3
from metatrain.utils.architectures import check_architecture_options
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import (
    get_energy_target_info,
)
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from . import DEFAULT_HYPERS, MODEL_HYPERS


def test_prediction():
    """Tests the basic functionality of the forward pass of the model."""
    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "energy": get_energy_target_info({"quantity": "energy", "unit": "eV"})
        },
    )

    model = DPA3(MODEL_HYPERS, dataset_info).to("cpu")

    system = System(
        types=torch.tensor([6, 6]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )
    system = get_system_with_neighbor_lists(system, model.requested_neighbor_lists())
    outputs = {"energy": ModelOutput(per_atom=False)}
    model([system, system], outputs)


def test_dpa3_padding():
    """Tests that the model predicts the same energy independently of the
    padding size."""

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "energy": get_energy_target_info({"quantity": "energy", "unit": "eV"})
        },
    )

    model = DPA3(MODEL_HYPERS, dataset_info).to("cpu")

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


def test_prediction_subset_elements():
    """Tests that the model can predict on a subset of the elements it was trained
    on."""

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "energy": get_energy_target_info({"quantity": "energy", "unit": "eV"})
        },
    )

    model = DPA3(MODEL_HYPERS, dataset_info).to("cpu")

    system = System(
        types=torch.tensor([6, 6]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )
    system = get_system_with_neighbor_lists(system, model.requested_neighbor_lists())
    model(
        [system],
        {"energy": model.outputs["energy"]},
    )


def test_prediction_subset_atoms():
    """Tests that the model can predict on a subset
    of the atoms in a system."""

    # we need float64 for this test, then we will change it back at the end
    default_dtype_before = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "energy": get_energy_target_info({"quantity": "energy", "unit": "eV"})
        },
    )

    model = DPA3(MODEL_HYPERS, dataset_info).to("cpu")

    # Since we don't yet support atomic predictions, we will test this by
    # predicting on a system with two monomers at a large distance

    system_monomer = System(
        types=torch.tensor([7, 8, 8]),
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]],
        ),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )
    system_monomer = get_system_with_neighbor_lists(
        system_monomer, model.requested_neighbor_lists()
    )

    energy_monomer = model(
        [system_monomer],
        {"energy": ModelOutput(per_atom=False)},
    )

    system_far_away_dimer = System(
        types=torch.tensor([7, 7, 8, 8, 8, 8]),
        positions=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 50.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 2.0],
                [0.0, 51.0, 0.0],
                [0.0, 42.0, 0.0],
            ],
        ),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )
    system_far_away_dimer = get_system_with_neighbor_lists(
        system_far_away_dimer, model.requested_neighbor_lists()
    )

    selection_labels = mts.Labels(
        names=["system", "atom"],
        values=torch.tensor([[0, 0], [0, 2], [0, 3]]),
    )

    energy_dimer = model(
        [system_far_away_dimer],
        {"energy": ModelOutput(per_atom=False)},
    )

    energy_monomer_in_dimer = model(
        [system_far_away_dimer],
        {"energy": ModelOutput(per_atom=False)},
        selected_atoms=selection_labels,
    )

    assert not mts.allclose(energy_monomer["energy"], energy_dimer["energy"])

    assert mts.allclose(
        energy_monomer["energy"], energy_monomer_in_dimer["energy"], atol=1e-6
    )

    torch.set_default_dtype(default_dtype_before)


def test_output_per_atom():
    """Tests that the model can output per-atom quantities."""
    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "energy": get_energy_target_info({"quantity": "energy", "unit": "eV"})
        },
    )

    model = DPA3(MODEL_HYPERS, dataset_info).to("cpu")

    system = System(
        types=torch.tensor([6, 1, 8, 7]),
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0]],
        ),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )
    system = get_system_with_neighbor_lists(system, model.requested_neighbor_lists())

    outputs = model(
        [system],
        {"energy": model.outputs["energy"]},
    )

    assert outputs["energy"].block().samples.names == ["system", "atom"]
    assert outputs["energy"].block().values.shape == (4, 1)


def test_fixed_composition_weights():
    """Tests the correctness of the json schema for fixed_composition_weights"""

    hypers = DEFAULT_HYPERS.copy()
    hypers["training"]["fixed_composition_weights"] = {
        "energy": {
            1: 1.0,
            6: 0.0,
            7: 0.0,
            8: 0.0,
            9: 3000.0,
        }
    }
    hypers = OmegaConf.create(hypers)
    check_architecture_options(
        name="experimental.dpa3", options=OmegaConf.to_container(hypers)
    )


def test_pet_single_atom():
    """Tests that the model predicts correctly on a single atom."""

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "energy": get_energy_target_info({"quantity": "energy", "unit": "eV"})
        },
    )
    model = DPA3(MODEL_HYPERS, dataset_info).to("cpu")

    system = System(
        types=torch.tensor([6]),
        positions=torch.tensor([[0.0, 0.0, 1.0]]),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )
    system = get_system_with_neighbor_lists(system, model.requested_neighbor_lists())
    outputs = {"energy": ModelOutput(per_atom=False)}
    model([system], outputs)
