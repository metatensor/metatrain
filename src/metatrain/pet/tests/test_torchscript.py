import copy

import torch

from metatrain.pet import PET as WrappedPET
from metatrain.pet.modules.hypers import Hypers
from metatrain.pet.modules.pet import PET
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_energy_target_info


DEFAULT_HYPERS = get_default_hypers("pet")


def test_torchscript():
    """Tests that the model can be jitted."""

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={"energy": get_energy_target_info({"unit": "eV"})},
    )
    model = WrappedPET(DEFAULT_HYPERS["model"], dataset_info)
    ARCHITECTURAL_HYPERS = Hypers(model.hypers)
    raw_pet = PET(ARCHITECTURAL_HYPERS, 0.0, len(model.atomic_types))
    model.set_trained_model(raw_pet)
    torch.jit.script(model)


def test_torchscript_save_load(tmpdir):
    """Tests that the model can be jitted and saved."""

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={"energy": get_energy_target_info({"unit": "eV"})},
    )
    model = WrappedPET(DEFAULT_HYPERS["model"], dataset_info)
    ARCHITECTURAL_HYPERS = Hypers(model.hypers)
    raw_pet = PET(ARCHITECTURAL_HYPERS, 0.0, len(model.atomic_types))
    model.set_trained_model(raw_pet)
    torch.jit.script(model)
    with tmpdir.as_cwd():
        torch.jit.save(torch.jit.script(model), "pet.pt")
        torch.jit.load("pet.pt")


def test_torchscript_integers():
    """Tests that the model can be jitted when some float
    parameters are instead supplied as integers."""

    new_hypers = copy.deepcopy(DEFAULT_HYPERS["model"])
    new_hypers["R_CUT"] = 5
    new_hypers["CUTOFF_DELTA"] = 1
    new_hypers["RESIDUAL_FACTOR"] = 1

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={"energy": get_energy_target_info({"unit": "eV"})},
    )
    model = WrappedPET(new_hypers, dataset_info)
    ARCHITECTURAL_HYPERS = Hypers(model.hypers)
    raw_pet = PET(ARCHITECTURAL_HYPERS, 0.0, len(model.atomic_types))
    model.set_trained_model(raw_pet)
    torch.jit.script(model)
