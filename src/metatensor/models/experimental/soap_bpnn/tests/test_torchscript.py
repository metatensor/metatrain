import copy

import ase
import torch
from metatensor.torch.atomistic import ModelCapabilities, ModelOutput, systems_to_torch

from metatensor.models.experimental.soap_bpnn import DEFAULT_HYPERS, Model


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
    soap_bpnn = Model(capabilities, DEFAULT_HYPERS["model"])
    soap_bpnn = torch.jit.script(soap_bpnn)

    system = ase.Atoms(
        "OHCN",
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0]],
    )
    soap_bpnn(
        [systems_to_torch(system)],
        {"energy": soap_bpnn.capabilities.outputs["energy"]},
    )


def test_torchscript_with_identity():
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
    hypers = copy.deepcopy(DEFAULT_HYPERS["model"])
    hypers["bpnn"]["layernorm"] = False
    soap_bpnn = Model(capabilities, hypers)
    soap_bpnn = torch.jit.script(soap_bpnn)

    system = ase.Atoms(
        "OHCN",
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0]],
    )
    soap_bpnn(
        [systems_to_torch(system)],
        {"energy": soap_bpnn.capabilities.outputs["energy"]},
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
    soap_bpnn = Model(capabilities, DEFAULT_HYPERS["model"])
    torch.jit.save(
        torch.jit.script(soap_bpnn),
        "soap_bpnn.pt",
    )
