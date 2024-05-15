import torch
from metatensor.torch.atomistic import ModelCapabilities, ModelOutput

from metatensor.models.experimental.pet import DEFAULT_HYPERS, Model


def test_torchscript():
    """Tests that the model can be jitted."""

    capabilities = ModelCapabilities(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        outputs={
            "energy": ModelOutput(
                quantity="energy",
                unit="eV",
            )
        },
        interaction_range=DEFAULT_HYPERS["ARCHITECTURAL_HYPERS"]["N_GNN_LAYERS"]
        * DEFAULT_HYPERS["ARCHITECTURAL_HYPERS"]["R_CUT"],
    )
    pet = Model(capabilities, DEFAULT_HYPERS["ARCHITECTURAL_HYPERS"])
    torch.jit.script(pet)


def test_torchscript_save():
    """Tests that the model can be jitted and saved."""

    capabilities = ModelCapabilities(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        outputs={
            "energy": ModelOutput(
                quantity="energy",
                unit="eV",
            )
        },
        interaction_range=DEFAULT_HYPERS["ARCHITECTURAL_HYPERS"]["N_GNN_LAYERS"]
        * DEFAULT_HYPERS["ARCHITECTURAL_HYPERS"]["R_CUT"],
    )
    pet = Model(capabilities, DEFAULT_HYPERS["ARCHITECTURAL_HYPERS"])
    torch.jit.save(
        torch.jit.script(pet),
        "pet.pt",
    )
