import ase
import pytest
import torch
from metatensor.torch.atomistic import (
    MetatensorAtomisticModel,
    ModelCapabilities,
    ModelEvaluationOptions,
    ModelMetadata,
    ModelOutput,
    NeighborListOptions,
    systems_to_torch,
)
from pet.data_preparation import get_pyg_graphs
from pet.hypers import Hypers

from metatensor.models.experimental.pet import DEFAULT_HYPERS, Model
from metatensor.models.experimental.pet.utils import systems_to_batch_dict
from metatensor.models.utils.neighbor_lists import get_system_with_neighbor_lists

from . import DATASET_PATH


def check_batch_dict_consistency(ref_batch_dict, trial_batch_dict):
    ref_mask = ref_batch_dict["mask"]
    trial_mask = trial_batch_dict["mask"]
    assert torch.all(ref_mask == trial_mask)
    mask = ref_mask == False
    for key in ref_batch_dict:
        if key in ("central_species", "x", "mask", "nums", "batch"):
            assert torch.all(ref_batch_dict[key] == trial_batch_dict[key])
        else:
            assert torch.all(ref_batch_dict[key][mask] == trial_batch_dict[key][mask])


@pytest.mark.parametrize("cutoff", [5.0])
def test_batch_dicts_compatibility(cutoff):
    """Tests that the batch dict computed with internal MTM routines
    is consitent with PET implementation."""

    structure = ase.io.read(DATASET_PATH)
    all_species = sorted(list(set(structure.numbers)))
    system = systems_to_torch(structure)
    options = NeighborListOptions(cutoff=cutoff, full_list=True)
    system = get_system_with_neighbor_lists(system, [options])

    ARCHITECTURAL_HYPERS = Hypers(DEFAULT_HYPERS["ARCHITECTURAL_HYPERS"])
    ref_batch_dict = get_pyg_graphs(
        [structure],
        all_species,
        cutoff,
        ARCHITECTURAL_HYPERS.USE_ADDITIONAL_SCALAR_ATTRIBUTES,
        ARCHITECTURAL_HYPERS.USE_LONG_RANGE,
        ARCHITECTURAL_HYPERS.K_CUT,
    )[0]

    trial_batch_dict = systems_to_batch_dict([system], options, all_species, None)
    check_batch_dict_consistency(ref_batch_dict, trial_batch_dict)


@pytest.mark.parametrize("cutoff", [0.25, 5.0])
def test_predictions_compatibility(cutoff):
    """Tests that predictions of the MTM implemetation of PET
    are consistent with the predictions of the original PET implementation."""

    structure = ase.io.read(DATASET_PATH)
    all_species = sorted(list(set(structure.numbers)))

    capabilities = ModelCapabilities(
        length_unit="Angstrom",
        atomic_types=all_species,
        outputs={
            "energy": ModelOutput(
                quantity="energy",
                unit="eV",
            )
        },
        supported_devices=["cuda", "cpu"],
        interaction_range=DEFAULT_HYPERS["ARCHITECTURAL_HYPERS"]["N_GNN_LAYERS"]
        * DEFAULT_HYPERS["ARCHITECTURAL_HYPERS"]["R_CUT"],
        dtype="float32",
    )
    hypers = DEFAULT_HYPERS["ARCHITECTURAL_HYPERS"]
    hypers["R_CUT"] = cutoff
    model = Model(capabilities, hypers)
    # structure = ase.Atoms("O2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    system = systems_to_torch(structure)

    options = NeighborListOptions(cutoff=cutoff, full_list=True)

    system = get_system_with_neighbor_lists(system, [options])

    evaluation_options = ModelEvaluationOptions(
        length_unit=capabilities.length_unit,
        outputs=capabilities.outputs,
    )

    model = MetatensorAtomisticModel(model.eval(), ModelMetadata(), model.capabilities)
    mtm_pet_prediction = (
        model(
            [system],
            evaluation_options,
            check_consistency=False,
        )["energy"]
        .block()
        .values
    )

    ARCHITECTURAL_HYPERS = Hypers(DEFAULT_HYPERS["ARCHITECTURAL_HYPERS"])
    batch = get_pyg_graphs(
        [structure],
        all_species,
        cutoff,
        ARCHITECTURAL_HYPERS.USE_ADDITIONAL_SCALAR_ATTRIBUTES,
        ARCHITECTURAL_HYPERS.USE_LONG_RANGE,
        ARCHITECTURAL_HYPERS.K_CUT,
    )[0]

    batch_dict = {
        "x": batch.x,
        "central_species": batch.central_species,
        "neighbor_species": batch.neighbor_species,
        "mask": batch.mask,
        "batch": torch.tensor([0] * len(batch.central_species)),
        "nums": batch.nums,
        "neighbors_index": batch.neighbors_index.transpose(0, 1),
        "neighbors_pos": batch.neighbors_pos,
    }

    pet = model._module.pet

    pet_prediction = pet.forward(batch_dict)

    assert torch.allclose(
        mtm_pet_prediction,
        pet_prediction.sum(dim=0),
    )
