import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from metatensor.learn.data import DataLoader
from metatensor.learn.data.dataset import _BaseDataset
from metatensor.torch.atomistic import ModelCapabilities
from pet.hypers import Hypers
from pet.pet import PET
from pet.train_model import fit_pet

from ...utils.data import collate_fn
from ...utils.data.system_to_ase import system_to_ase
from .model import DEFAULT_HYPERS, Model


logger = logging.getLogger(__name__)


def train(
    train_datasets: List[Union[_BaseDataset, torch.utils.data.Subset]],
    validation_datasets: List[Union[_BaseDataset, torch.utils.data.Subset]],
    requested_capabilities: ModelCapabilities,
    hypers: Dict = DEFAULT_HYPERS,
    continue_from: Optional[str] = None,
    output_dir: str = ".",
    device_str: str = "cpu",
):
    if torch.get_default_dtype() != torch.float32:
        raise ValueError("PET only supports float32")
    if device_str != "cuda" and device_str != "gpu":
        raise ValueError("PET only supports cuda (gpu) training")
    if len(requested_capabilities.outputs) != 1:
        raise ValueError("PET only supports a single output")
    target_name = next(iter(requested_capabilities.outputs.keys()))
    if requested_capabilities.outputs[target_name].quantity != "energy":
        raise ValueError("PET only supports energies as output")
    if requested_capabilities.outputs[target_name].per_atom:
        raise ValueError("PET does not support per-atom energies")
    if len(train_datasets) != 1:
        raise ValueError("PET only supports a single training dataset")
    if len(validation_datasets) != 1:
        raise ValueError("PET only supports a single validation dataset")

    if device_str == "gpu":
        device_str = "cuda"

    if continue_from is not None:
        hypers["FITTING_SCHEME"]["MODEL_TO_START_WITH"] = continue_from

    train_dataset = train_datasets[0]
    validation_dataset = validation_datasets[0]

    # dummy dataloaders due to https://github.com/lab-cosmo/metatensor/issues/521
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
    )
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # only energies or energies and forces?
    do_forces = next(iter(train_dataset))[1].block().has_gradient("positions")
    all_species = requested_capabilities.species
    if not do_forces:
        hypers["MLIP_SETTINGS"]["USE_FORCES"] = False

    ase_train_dataset = []
    for (system,), targets in train_dataloader:
        ase_atoms = system_to_ase(system)
        ase_atoms.info["energy"] = float(
            targets[target_name].block().values.squeeze(-1).detach().cpu().numpy()
        )
        if do_forces:
            ase_atoms.arrays["forces"] = (
                -targets[target_name]
                .block()
                .gradient("positions")
                .values.squeeze(-1)
                .detach()
                .cpu()
                .numpy()
            )
        ase_train_dataset.append(ase_atoms)

    ase_validation_dataset = []
    for (system,), targets in validation_dataloader:
        ase_atoms = system_to_ase(system)
        ase_atoms.info["energy"] = float(
            targets[target_name].block().values.squeeze(-1).detach().cpu().numpy()
        )
        if do_forces:
            ase_atoms.arrays["forces"] = (
                -targets[target_name]
                .block()
                .gradient("positions")
                .values.squeeze(-1)
                .detach()
                .cpu()
                .numpy()
            )
        ase_validation_dataset.append(ase_atoms)

    fit_pet(
        ase_train_dataset, ase_validation_dataset, hypers, "pet", device_str, output_dir
    )

    if do_forces:
        load_path = Path(output_dir) / "pet" / "best_val_rmse_forces_model_state_dict"
    else:
        load_path = Path(output_dir) / "pet" / "best_val_rmse_energies_model_state_dict"

    state_dict = torch.load(load_path)

    ARCHITECTURAL_HYPERS = Hypers(hypers["ARCHITECTURAL_HYPERS"])
    ARCHITECTURAL_HYPERS.D_OUTPUT = 1  # energy is a single scalar
    ARCHITECTURAL_HYPERS.TARGET_TYPE = "structural"  # energy is structural property
    ARCHITECTURAL_HYPERS.TARGET_AGGREGATION = (
        "sum"  # energy is a sum of atomic energies
    )

    raw_pet = PET(ARCHITECTURAL_HYPERS, 0.0, len(all_species))

    new_state_dict = {}
    for name, value in state_dict.items():
        name = name.replace("model.pet_model.", "")
        new_state_dict[name] = value

    raw_pet.load_state_dict(new_state_dict)

    model = Model(requested_capabilities, ARCHITECTURAL_HYPERS)

    model.set_trained_model(raw_pet)

    return model
