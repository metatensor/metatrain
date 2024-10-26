import copy

import ase.io
import torch
from metatensor.torch.atomistic import ModelEvaluationOptions, systems_to_torch

from metatrain.experimental.alchemical_model import AlchemicalModel
from metatrain.utils.data import DatasetInfo, TargetInfo
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)
from metatrain.utils.testing import energy_layout

from . import DATASET_PATH, MODEL_HYPERS


def test_rotational_invariance():
    """Tests that the model is rotationally invariant."""

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "energy": TargetInfo(quantity="energy", unit="eV", layout=energy_layout)
        },
    )
    model = AlchemicalModel(MODEL_HYPERS, dataset_info)

    system = ase.io.read(DATASET_PATH)
    original_system = copy.deepcopy(system)
    original_system = systems_to_torch(original_system)
    requested_neighbor_lists = get_requested_neighbor_lists(model)
    original_system = get_system_with_neighbor_lists(
        original_system, requested_neighbor_lists
    )

    system.rotate(48, "y")
    system = systems_to_torch(system)
    requested_neighbor_lists = get_requested_neighbor_lists(model)
    system = get_system_with_neighbor_lists(system, requested_neighbor_lists)

    evaluation_options = ModelEvaluationOptions(
        length_unit=dataset_info.length_unit,
        outputs=model.outputs,
    )

    exported = model.export()

    original_output = exported(
        [original_system],
        evaluation_options,
        check_consistency=True,
    )
    rotated_output = exported([system], evaluation_options, check_consistency=True)

    torch.testing.assert_close(
        original_output["energy"].block().values,
        rotated_output["energy"].block().values,
    )
