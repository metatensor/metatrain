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

from metatrain.pet import PET as WrappedPET
from metatrain.pet.modules.data_preparation import get_pyg_graphs
from metatrain.pet.modules.hypers import Hypers
from metatrain.pet.modules.pet import PET
from metatrain.pet.utils import systems_to_batch_dict
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.readers.ase import read
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from . import DATASET_PATH


DEFAULT_HYPERS = get_default_hypers("pet")


def check_batch_dict_consistency(ref_batch, trial_batch):
    ref_mask = ref_batch["mask"]
    trial_mask = trial_batch["mask"]
    assert torch.all(ref_mask == trial_mask)
    mask = ref_mask is False

    for key in ref_batch:
        if key == "x":
            torch.testing.assert_close(
                ref_batch["x"].flatten().sort()[0],
                trial_batch["x"].flatten().sort()[0],
                rtol=1e-6,
                atol=1e-5,
            )
        elif key in ("central_species", "mask", "nums", "batch"):
            assert torch.all(ref_batch[key] == trial_batch[key])
        else:
            ref_unique, ref_counts = ref_batch[key][mask].unique(return_counts=True)
            trial_unique, trial_counts = trial_batch[key][mask].unique(
                return_counts=True
            )
            assert torch.all(ref_unique == trial_unique)
            assert torch.all(ref_counts == trial_counts)


@pytest.mark.parametrize("cutoff", [0.25, 5.0])
def test_batch_dicts_compatibility(cutoff):
    """Tests that the batch dict computed with internal MTM routines
    is consitent with PET implementation."""

    structure = read(DATASET_PATH)
    atomic_types = sorted(set(structure.numbers))
    system = systems_to_torch(structure)
    options = NeighborListOptions(cutoff=cutoff, full_list=True, strict=True)
    system = get_system_with_neighbor_lists(system, [options])

    ARCHITECTURAL_HYPERS = Hypers(DEFAULT_HYPERS["model"])
    batch = get_pyg_graphs(
        [structure],
        atomic_types,
        cutoff,
        ARCHITECTURAL_HYPERS.USE_ADDITIONAL_SCALAR_ATTRIBUTES,
        ARCHITECTURAL_HYPERS.USE_LONG_RANGE,
        ARCHITECTURAL_HYPERS.K_CUT,
        ARCHITECTURAL_HYPERS.N_TARGETS > 1,
        ARCHITECTURAL_HYPERS.TARGET_INDEX_KEY,
    )[0]
    ref_batch_dict = {
        "x": batch.x,
        "central_species": batch.central_species,
        "neighbor_species": batch.neighbor_species,
        "mask": batch.mask,
        "batch": torch.tensor([0] * len(batch.central_species)),
        "nums": batch.nums,
        "neighbors_index": batch.neighbors_index.transpose(0, 1),
        "neighbors_pos": batch.neighbors_pos,
    }
    trial_batch_dict = systems_to_batch_dict([system], options, atomic_types, None)
    check_batch_dict_consistency(ref_batch_dict, trial_batch_dict)


@pytest.mark.parametrize("cutoff", [0.25, 5.0])
def test_predictions_compatibility(cutoff):
    """Tests that predictions of the MTM implemetation of PET
    are consistent with the predictions of the original PET implementation."""

    structure = read(DATASET_PATH)

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=structure.numbers,
        targets={"energy": get_energy_target_info({"unit": "eV"})},
    )
    capabilities = ModelCapabilities(
        length_unit="Angstrom",
        atomic_types=dataset_info.atomic_types,
        outputs={
            "energy": ModelOutput(
                quantity="energy",
                unit="eV",
            )
        },
        interaction_range=DEFAULT_HYPERS["model"]["N_GNN_LAYERS"]
        * DEFAULT_HYPERS["model"]["R_CUT"],
        dtype="float32",
        supported_devices=["cpu", "cuda"],
    )
    hypers = DEFAULT_HYPERS["model"]
    hypers["R_CUT"] = cutoff
    model = WrappedPET(DEFAULT_HYPERS["model"], dataset_info)
    ARCHITECTURAL_HYPERS = Hypers(model.hypers)
    raw_pet = PET(ARCHITECTURAL_HYPERS, 0.0, len(dataset_info.atomic_types))
    model.set_trained_model(raw_pet)

    system = systems_to_torch(structure)
    options = NeighborListOptions(cutoff=cutoff, full_list=True, strict=True)
    system = get_system_with_neighbor_lists(system, [options])

    evaluation_options = ModelEvaluationOptions(
        length_unit=capabilities.length_unit,
        outputs=capabilities.outputs,
    )

    model = MetatensorAtomisticModel(model.eval(), ModelMetadata(), capabilities)
    mtm_pet_prediction = (
        model(
            [system],
            evaluation_options,
            check_consistency=False,
        )["energy"]
        .block()
        .values
    )

    ARCHITECTURAL_HYPERS = Hypers(DEFAULT_HYPERS["model"])
    batch = get_pyg_graphs(
        [structure],
        dataset_info.atomic_types,
        cutoff,
        ARCHITECTURAL_HYPERS.USE_ADDITIONAL_SCALAR_ATTRIBUTES,
        ARCHITECTURAL_HYPERS.USE_LONG_RANGE,
        ARCHITECTURAL_HYPERS.K_CUT,
        ARCHITECTURAL_HYPERS.N_TARGETS > 1,
        ARCHITECTURAL_HYPERS.TARGET_INDEX_KEY,
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

    pet = model.module.pet

    pet_prediction = pet.forward(batch_dict)["prediction"]

    torch.testing.assert_close(
        mtm_pet_prediction, pet_prediction.sum(dim=0, keepdim=True)
    )
