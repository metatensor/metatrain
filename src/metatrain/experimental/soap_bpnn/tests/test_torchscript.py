import copy

import torch
from metatensor.torch.atomistic import System

from metatrain.experimental.soap_bpnn import SoapBpnn
from metatrain.utils.data import DatasetInfo, TargetInfo, TargetInfoDict

from . import MODEL_HYPERS


def test_torchscript():
    """Tests that the model can be jitted."""

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types={1, 6, 7, 8},
        targets=TargetInfoDict(energy=TargetInfo(quantity="energy", unit="eV")),
    )
    model = SoapBpnn(MODEL_HYPERS, dataset_info)
    model = torch.jit.script(model)

    system = System(
        types=[6, 1, 8, 7],
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0]]
        ),
        cell=torch.zeros(3, 3),
    )
    model(
        [system],
        {"energy": model.outputs["energy"]},
    )


def test_torchscript_with_identity():
    """Tests that the model can be jitted."""

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types={1, 6, 7, 8},
        targets=TargetInfoDict(energy=TargetInfo(quantity="energy", unit="eV")),
    )
    hypers = copy.deepcopy(MODEL_HYPERS)
    hypers["bpnn"]["layernorm"] = False
    model = SoapBpnn(hypers, dataset_info)
    model = torch.jit.script(model)

    system = System(
        types=[6, 1, 8, 7],
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0]]
        ),
        cell=torch.zeros(3, 3),
    )
    model(
        [system],
        {"energy": model.outputs["energy"]},
    )


def test_torchscript_save_load():
    """Tests that the model can be jitted and saved."""

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types={1, 6, 7, 8},
        targets=TargetInfoDict(energy=TargetInfo(quantity="energy", unit="eV")),
    )
    model = SoapBpnn(MODEL_HYPERS, dataset_info)
    torch.jit.save(
        torch.jit.script(model),
        "model.pt",
    )
    torch.jit.load("model.pt")
