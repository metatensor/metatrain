import copy

import ase.io
import torch
from metatensor.torch.atomistic import systems_to_torch

from metatrain.experimental.phace import PhACE
from metatrain.utils.data import DatasetInfo, TargetInfo, TargetInfoDict
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from . import DATASET_PATH, MODEL_HYPERS


def test_rotational_invariance():
    """Tests that the model is rotationally invariant."""

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets=TargetInfoDict(energy=TargetInfo(quantity="energy", unit="eV")),
    )
    model = PhACE(MODEL_HYPERS, dataset_info)

    system = ase.io.read(DATASET_PATH)
    original_system = copy.deepcopy(system)
    system.rotate(48, "y")

    original_system = systems_to_torch(original_system).to(torch.float64)
    system = systems_to_torch(system).to(torch.float64)

    nls = model.requested_neighbor_lists()
    original_system = get_system_with_neighbor_lists(original_system, nls)
    system = get_system_with_neighbor_lists(system, nls)

    model = torch.jit.script(model).to(torch.float64)

    original_output = model(
        [original_system, system],
        {"energy": model.outputs["energy"]},
    )
    rotated_output = model(
        [system, original_system],
        {"energy": model.outputs["energy"]},
    )

    torch.testing.assert_close(
        original_output["energy"].block().values,
        rotated_output["energy"].block().values,
    )
