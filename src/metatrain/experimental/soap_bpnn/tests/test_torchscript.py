import copy

import torch
from metatensor.torch.atomistic import System

from metatrain.experimental.soap_bpnn import SoapBpnn
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_energy_target_info

from . import MODEL_HYPERS


def test_torchscript():
    """Tests that the model can be jitted."""

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={"energy": get_energy_target_info({"unit": "eV"})},
    )
    model = SoapBpnn(MODEL_HYPERS, dataset_info)
    model = torch.jit.script(model)

    system = System(
        types=torch.tensor([6, 1, 8, 7]),
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0]]
        ),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )
    model(
        [system],
        {"energy": model.outputs["energy"]},
    )


def test_torchscript_with_identity():
    """Tests that the model can be jitted."""

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={"energy": get_energy_target_info({"unit": "eV"})},
    )
    hypers = copy.deepcopy(MODEL_HYPERS)
    hypers["bpnn"]["layernorm"] = False
    model = SoapBpnn(hypers, dataset_info)
    model = torch.jit.script(model)

    system = System(
        types=torch.tensor([6, 1, 8, 7]),
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0]]
        ),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )
    model(
        [system],
        {"energy": model.outputs["energy"]},
    )


def test_torchscript_save_load():
    """Tests that the model can be jitted and saved."""

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={"energy": get_energy_target_info({"unit": "eV"})},
    )
    model = SoapBpnn(MODEL_HYPERS, dataset_info)
    torch.jit.save(
        torch.jit.script(model),
        "model.pt",
    )
    torch.jit.load("model.pt")


def test_torchscript_integers():
    """Tests that the model can be jitted when some float
    parameters are instead supplied as integers."""

    new_hypers = copy.deepcopy(MODEL_HYPERS)
    new_hypers["soap"]["cutoff"] = 5
    new_hypers["soap"]["atomic_gaussian_width"] = 1
    new_hypers["soap"]["center_atom_weight"] = 1
    new_hypers["soap"]["cutoff_function"]["ShiftedCosine"]["width"] = 1
    new_hypers["soap"]["radial_scaling"]["Willatt2018"]["rate"] = 1
    new_hypers["soap"]["radial_scaling"]["Willatt2018"]["scale"] = 2
    new_hypers["soap"]["radial_scaling"]["Willatt2018"]["exponent"] = 7

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={"energy": get_energy_target_info({"unit": "eV"})},
    )
    model = SoapBpnn(new_hypers, dataset_info)
    model = torch.jit.script(model)

    system = System(
        types=torch.tensor([6, 1, 8, 7]),
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0]]
        ),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )
    model(
        [system],
        {"energy": model.outputs["energy"]},
    )
