import logging
from pathlib import Path
from typing import List, Union

import torch
from metatensor.learn.data import DataLoader
from pet.hypers import Hypers
from pet.pet import PET
from pet.train_model import fit_pet

from ...utils.data import Dataset, check_datasets, collate_fn
from ...utils.data.system_to_ase import system_to_ase
from . import PET as WrappedPET


logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, train_hypers):
        self.hypers = {"FITTING_SCHEME": train_hypers}

    def train(
        self,
        model: WrappedPET,
        devices: List[torch.device],
        train_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        validation_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        checkpoint_dir: str,
    ):
        if len(train_datasets) != 1:
            raise ValueError("PET only supports a single training dataset")
        if len(validation_datasets) != 1:
            raise ValueError("PET only supports a single validation dataset")

        if model.checkpoint_path is not None:
            self.hypers["FITTING_SCHEME"]["MODEL_TO_START_WITH"] = model.checkpoint_path

        logger.info("Checking datasets for consistency")
        check_datasets(train_datasets, validation_datasets)

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

        # are we fitting on only energies or energies and forces?
        target_name = model.target_name
        do_forces = (
            next(iter(train_dataset))[target_name].block().has_gradient("positions")
        )

        # set model hypers
        self.hypers["ARCHITECTURAL_HYPERS"] = model.hypers
        dtype = train_datasets[0][0]["system"].positions.dtype
        if dtype != torch.float32:
            raise ValueError("PET only supports float32 as dtype")
        self.hypers["ARCHITECTURAL_HYPERS"]["DTYPE"] = "float32"

        # set MLIP_SETTINGS
        self.hypers["MLIP_SETTINGS"] = {
            "ENERGY_KEY": "energy",
            "FORCES_KEY": "forces",
            "USE_ENERGIES": True,
            "USE_FORCES": do_forces,
        }

        # set PET utility flags
        self.hypers["UTILITY_FLAGS"] = {
            "CALCULATION_TYPE": None,
        }

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

        device = devices[0]  # only one device, as we don't support multi-gpu for now

        fit_pet(
            ase_train_dataset,
            ase_validation_dataset,
            self.hypers,
            "pet",
            device,
            checkpoint_dir,
        )

        if do_forces:
            load_path = (
                Path(checkpoint_dir) / "pet" / "best_val_rmse_forces_model_state_dict"
            )
        else:
            load_path = (
                Path(checkpoint_dir) / "pet" / "best_val_rmse_energies_model_state_dict"
            )

        state_dict = torch.load(load_path)

        ARCHITECTURAL_HYPERS = Hypers(model.hypers)
        raw_pet = PET(ARCHITECTURAL_HYPERS, 0.0, len(model.species))

        new_state_dict = {}
        for name, value in state_dict.items():
            name = name.replace("model.pet_model.", "")
            new_state_dict[name] = value

        raw_pet.load_state_dict(new_state_dict)

        model.set_trained_model(raw_pet)

        return model
