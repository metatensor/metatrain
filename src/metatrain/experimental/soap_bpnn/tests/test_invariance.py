import copy

import ase.io
import torch
from metatensor.torch.atomistic import systems_to_torch

from metatrain.experimental.soap_bpnn import SoapBpnn
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_energy_target_info

from . import DATASET_PATH, MODEL_HYPERS


def test_rotational_invariance():
    """Tests that the model is rotationally invariant."""

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={"energy": get_energy_target_info({"unit": "eV"})},
    )
    model = SoapBpnn(MODEL_HYPERS, dataset_info)

    system = ase.io.read(DATASET_PATH)
    original_system = copy.deepcopy(system)
    system.rotate(48, "y")

    original_output = model(
        [systems_to_torch(original_system)],
        {"energy": model.outputs["energy"]},
    )
    rotated_output = model(
        [systems_to_torch(system)],
        {"energy": model.outputs["energy"]},
    )

    torch.testing.assert_close(
        original_output["energy"].block().values,
        rotated_output["energy"].block().values,
    )
