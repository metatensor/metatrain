import torch  # noqa: E402
from metatensor.torch.atomistic import ModelCapabilities, ModelOutput  # noqa: E402

from metatensor.models.experimental.alchemical_model import (  # noqa: E402
    DEFAULT_HYPERS,
    Model,
)


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
        interaction_range=DEFAULT_HYPERS["model"]["soap"]["cutoff"],
        dtype="float32",
    )
    alchemical_model = Model(capabilities, DEFAULT_HYPERS["model"])
    torch.jit.script(
        alchemical_model, {"energy": alchemical_model.capabilities.outputs["energy"]}
    )


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
        interaction_range=DEFAULT_HYPERS["model"]["soap"]["cutoff"],
        dtype="float32",
    )
    alchemical_model = Model(capabilities, DEFAULT_HYPERS["model"])
    torch.jit.save(
        torch.jit.script(
            alchemical_model,
            {"energy": alchemical_model.capabilities.outputs["energy"]},
        ),
        "alchemical_model.pt",
    )
