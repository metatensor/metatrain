import ase
import torch
from metatensor.torch.atomistic import (
    MetatensorAtomisticModel,
    ModelCapabilities,
    ModelEvaluationOptions,
    ModelMetadata,
    ModelOutput,
    NeighborsListOptions,
    systems_to_torch,
)

from metatensor.models.experimental.pet import DEFAULT_HYPERS, Model
from metatensor.models.utils.neighbors_lists import get_system_with_neighbors_lists

from pet.data_preparation import get_pyg_graphs
from pet.hypers import Hypers
import pytest


@pytest.mark.parametrize("cutoff", [0.25, 5.0])
def test_predictions_compatibility(cutoff):
    """Tests that the model runs without errors."""

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
    )
    hypers = DEFAULT_HYPERS["ARCHITECTURAL_HYPERS"]
    hypers["R_CUT"] = cutoff
    model = Model(capabilities, hypers)
    structure = ase.Atoms("O2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    system = systems_to_torch(structure)

    options = NeighborsListOptions(cutoff=cutoff, full_list=True)

    system = get_system_with_neighbors_lists(system, [options])

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
