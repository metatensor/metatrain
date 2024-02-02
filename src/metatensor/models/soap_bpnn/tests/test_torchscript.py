import os


# Execute the setup script which will make sum_over_samples saveable.
current_dir = os.path.dirname(__file__)
setup_path = os.path.join(
    current_dir, "..", "..", "..", "..", "..", "scripts", "hotfix_metatensor.py"
)
exec(open(setup_path).read())

import torch  # noqa: E402
from metatensor.torch.atomistic import ModelCapabilities, ModelOutput  # noqa: E402

from metatensor.models.soap_bpnn import DEFAULT_HYPERS, Model  # noqa: E402


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
