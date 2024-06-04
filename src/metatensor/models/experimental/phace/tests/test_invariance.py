import copy

import ase.io
import torch
from metatensor.torch.atomistic import ModelCapabilities, ModelOutput, systems_to_torch

from metatensor.models.experimental.soap_bpnn import DEFAULT_HYPERS, Model

from . import DATASET_PATH


def test_rotational_invariance():
    """Tests that the model is rotationally invariant."""

    capabilities = ModelCapabilities(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        outputs={
            "energy": ModelOutput(
                quantity="energy",
                unit="eV",
            )
        },
    )
    soap_bpnn = Model(capabilities, DEFAULT_HYPERS["model"])

    system = ase.io.read(DATASET_PATH)
    original_system = copy.deepcopy(system)
    system.rotate(48, "y")

    original_output = soap_bpnn(
        [systems_to_torch(original_system)],
        {"energy": soap_bpnn.capabilities.outputs["energy"]},
    )
    rotated_output = soap_bpnn(
        [systems_to_torch(system)],
        {"energy": soap_bpnn.capabilities.outputs["energy"]},
    )

    torch.testing.assert_close(
        original_output["energy"].block().values,
        rotated_output["energy"].block().values,
    )
