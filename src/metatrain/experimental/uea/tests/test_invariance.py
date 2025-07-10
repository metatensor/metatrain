import copy

import ase.io
import torch
from metatomic.torch import systems_to_torch

from metatrain.experimental.uea import UEA
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import (
    get_energy_target_info,
)
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from . import DATASET_PATH, MODEL_HYPERS


def test_rotational_invariance():
    """Tests that the model is rotationally invariant for a scalar target."""
    torch.set_default_dtype(torch.float64)  # make sure we have enough precision

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={"energy": get_energy_target_info({"unit": "eV"})},
    )
    model = UEA(MODEL_HYPERS, dataset_info)

    system = ase.io.read(DATASET_PATH)
    original_system = copy.deepcopy(system)
    system.rotate(48, "y")

    original_output = model(
        [
            get_system_with_neighbor_lists(
                systems_to_torch(original_system), model.requested_neighbor_lists()
            )
        ],
        {"energy": model.outputs["energy"]},
    )
    rotated_output = model(
        [
            get_system_with_neighbor_lists(
                systems_to_torch(system), model.requested_neighbor_lists()
            )
        ],
        {"energy": model.outputs["energy"]},
    )

    torch.testing.assert_close(
        original_output["energy"].block().values,
        rotated_output["energy"].block().values,
        atol=1e-5,
        rtol=1e-5,
    )

    torch.set_default_dtype(torch.float32)  # change back
