import copy

import torch

from metatrain.experimental.alchemical_model import AlchemicalModel
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

    model = AlchemicalModel(MODEL_HYPERS, dataset_info)
    torch.jit.script(model, {"energy": model.outputs["energy"]})


def test_torchscript_save_load():
    """Tests that the model can be jitted and saved."""

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={"energy": get_energy_target_info({"unit": "eV"})},
    )
    model = AlchemicalModel(MODEL_HYPERS, dataset_info)
    torch.jit.save(
        torch.jit.script(
            model,
            {"energy": model.outputs["energy"]},
        ),
        "alchemical_model.pt",
    )

    torch.jit.load("alchemical_model.pt")


def test_torchscript_integers():
    """Tests that the model can be jitted when some float
    parameters are instead supplied as integers."""

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={"energy": get_energy_target_info({"unit": "eV"})},
    )

    new_hypers = copy.deepcopy(MODEL_HYPERS)
    new_hypers["soap"]["cutoff"] = 5
    new_hypers["soap"]["basis_scale"] = 3

    model = AlchemicalModel(MODEL_HYPERS, dataset_info)
    torch.jit.script(model, {"energy": model.outputs["energy"]})
