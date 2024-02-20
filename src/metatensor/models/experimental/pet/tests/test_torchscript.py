import torch
from metatensor.torch.atomistic import ModelCapabilities, ModelOutput

from metatensor.models.experimental.pet import DEFAULT_HYPERS, Model


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
    model = Model(capabilities, DEFAULT_HYPERS["ARCHITECTURAL_HYPERS"]).to(
        torch.float64
    )
    torch.jit.script(model, {"energy": model.capabilities.outputs["energy"]})


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
    model = Model(capabilities, DEFAULT_HYPERS["ARCHITECTURAL_HYPERS"]).to(
        torch.float64
    )
    torch.jit.save(
        torch.jit.script(model, {"energy": model.capabilities.outputs["energy"]}),
        "model.pt",
    )
