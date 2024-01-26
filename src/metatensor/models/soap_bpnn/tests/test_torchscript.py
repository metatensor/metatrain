import os

import torch
from metatensor.torch.atomistic import ModelCapabilities, ModelOutput

from metatensor.models.soap_bpnn import DEFAULT_HYPERS, Model


def test_torchscript():
    """Tests that the model can be jitted."""

    capabilities = ModelCapabilities(
        length_unit="Angstrom",
        species=[1, 6, 7, 8],
        outputs={
            "energy": ModelOutput(
                quantity="energy",
                unit="eV",
            )
        },
    )
    soap_bpnn = Model(capabilities, DEFAULT_HYPERS["model"]).to(torch.float64)
    torch.jit.script(soap_bpnn, {"energy": soap_bpnn.capabilities.outputs["energy"]})


def test_torchscript_save():
    """Tests that the model can be jitted and saved."""

    # Execute the setup script which will make sum_over_samples saveable.
    current_dir = os.path.dirname(__file__)
    setup_path = os.path.join(
        current_dir, "..", "..", "..", "..", "..", "scripts", "setup.py"
    )
    exec(open(setup_path).read())

    capabilities = ModelCapabilities(
        length_unit="Angstrom",
        species=[1, 6, 7, 8],
        outputs={
            "energy": ModelOutput(
                quantity="energy",
                unit="eV",
            )
        },
    )
    soap_bpnn = Model(capabilities, DEFAULT_HYPERS["model"]).to(torch.float64)
    torch.jit.save(
        torch.jit.script(
            soap_bpnn, {"energy": soap_bpnn.capabilities.outputs["energy"]}
        ),
        "soap_bpnn.pt",
    )
