import torch

from metatrain.experimental.alchemical_model import AlchemicalModel
from metatrain.utils.data import DatasetInfo, TargetInfo
from metatrain.utils.testing import energy_layout

from . import MODEL_HYPERS


def test_torchscript():
    """Tests that the model can be jitted."""

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "energy": TargetInfo(quantity="energy", unit="eV", layout=energy_layout)
        },
    )

    model = AlchemicalModel(MODEL_HYPERS, dataset_info)
    torch.jit.script(model, {"energy": model.outputs["energy"]})


def test_torchscript_save_load():
    """Tests that the model can be jitted and saved."""

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "energy": TargetInfo(quantity="energy", unit="eV", layout=energy_layout)
        },
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
