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
from metatensor.models.utils.neighbor_lists import get_system_with_neighbor_lists


@pytest.mark.parametrize("cutoff", [0.25, 5.0])
def test_predictions_compatibility(cutoff):
    """Tests that predictions of the MTM implemetation of PET
    are consistent with the predictions of the original PET implementation."""

    all_species = [1, 6, 7, 8]

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
    structure = ase.Atoms("O2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
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
        "neighbor_index": batch.neighbor_index.transpose(0, 1),
        "neighbor_pos": batch.neighbor_pos,
    }

    pet = model._module.pet

    pet_prediction = pet.forward(batch_dict)

    assert torch.allclose(
        mtm_pet_prediction,
        pet_prediction.sum(dim=0),
    )
