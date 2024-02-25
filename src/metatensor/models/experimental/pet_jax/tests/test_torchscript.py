import torch
from metatensor.torch.atomistic import ModelCapabilities, ModelOutput

from metatensor.models.experimental.pet_jax import DEFAULT_HYPERS, Model


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
    pet = Model(
        capabilities, DEFAULT_HYPERS["model"], torch.tensor([0.1, 0.2, 0.3, 0.4])
    )
    torch.jit.script(pet, {"energy": pet.capabilities.outputs["energy"]})


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
    pet = Model(
        capabilities, DEFAULT_HYPERS["model"], torch.tensor([0.1, 0.2, 0.3, 0.4])
    )
    torch.jit.save(
        torch.jit.script(pet, {"energy": pet.capabilities.outputs["energy"]}),
        "pet.pt",
    )
