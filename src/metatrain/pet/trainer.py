import datetime
import logging
import os
import pickle
import time
import warnings
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from torch_geometric.nn import DataParallel
from torch.utils.data import DataLoader

from ..utils.data import Dataset, check_datasets
from ..utils.io import check_file_extension
from .model import PET
from .modules.analysis import adapt_hypers
from .modules.data_preparation import (
    get_corrected_energies,
    get_forces,
    get_pyg_graphs,
    get_self_contributions,
    update_pyg_graphs,
)
from ..utils.additive import remove_additive
from ..utils.data import CombinedDataLoader, Dataset, collate_fn

from .modules.hypers import Hypers, save_hypers
from .modules.pet import PET as RawPET
from ..utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)
from .modules.pet import FlagsWrapper, PETMLIPWrapper, PETUtilityWrapper
from .modules.utilities import (
    FullLogger,
    ModelKeeper,
    dtype2string,
    get_calc_names,
    get_data_loaders,
    get_loss,
    get_optimizer,
    get_rmse,
    log_epoch_stats,
    set_reproducibility,
    string2dtype,
)
from .utils import dataset_to_ase, load_raw_pet_model, update_hypers
from .utils.fine_tuning import LoRAWrapper
from .modules.augmentation import RotationalAugmenter
from torch.optim.lr_scheduler import LambdaLR


logger = logging.getLogger(__name__)


def get_scheduler(optimizer, hypers):
    def func_lr_scheduler(epoch):
        if epoch < hypers["EPOCHS_WARMUP"]:
            return epoch / hypers["EPOCHS_WARMUP"]
        delta = epoch - hypers["EPOCHS_WARMUP"]
        num_blocks = delta // hypers["SCHEDULER_STEP_SIZE"]
        return 0.5 ** (num_blocks)

    scheduler = LambdaLR(optimizer, func_lr_scheduler)
    return scheduler


class Trainer:
    def __init__(self, train_hypers):
        self.hypers = {"FITTING_SCHEME": train_hypers}
        self.pet_dir = None
        self.pet_trainer_state = None
        self.epoch = None
        self.best_metric = None
        self.best_model_state_dict = None

    def train(
        self,
        model: PET,
        dtype: torch.dtype,
        devices: List[torch.device],
        train_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        val_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        checkpoint_dir: str,
    ):
        assert dtype in PET.__supported_dtypes__
        device = devices[0]
        logger.info(f"Training on device {device} with dtype {dtype}")

        logger.info("Calculating neighbor lists for the datasets")
        requested_neighbor_lists = get_requested_neighbor_lists(model)
        for dataset in train_datasets + val_datasets:
            for i in range(len(dataset)):
                system = dataset[i]["system"]
                # The following line attaches the neighbors lists to the system,
                # and doesn't require to reassign the system to the dataset:
                _ = get_system_with_neighbor_lists(system, requested_neighbor_lists)

        # Move the model to the device and dtype:
        model.to(device=device, dtype=dtype)
        # The additive models of the SOAP-BPNN are always in float64 (to avoid
        # numerical errors in the composition weights, which can be very large).
        for additive_model in model.additive_models:
            additive_model.to(dtype=torch.float64)

        logger.info("Calculating composition weights")
        model.additive_models[0].train_model(  # this is the composition model
            train_datasets,
            model.additive_models[1:],
            self.hypers["fixed_composition_weights"],
        )

        if self.hypers["scale_targets"]:
            logger.info("Calculating scaling weights")
            model.scaler.train_model(
                train_datasets, model.additive_models, treat_as_additive=True
            )

        train_samplers = [None] * len(train_datasets)
        val_samplers = [None] * len(val_datasets)

        # Create dataloader for the training datasets:
        train_dataloaders = []
        for dataset, sampler in zip(train_datasets, train_samplers):
            train_dataloaders.append(
                DataLoader(
                    dataset=dataset,
                    batch_size=self.hypers["batch_size"],
                    sampler=sampler,
                    shuffle=(
                        sampler is None
                    ),  # the sampler takes care of this (if present)
                    drop_last=(
                        sampler is None
                    ),  # the sampler takes care of this (if present)
                    collate_fn=collate_fn,
                )
            )
        train_dataloader = CombinedDataLoader(train_dataloaders, shuffle=True)

        # Create dataloader for the validation datasets:
        val_dataloaders = []
        for dataset, sampler in zip(val_datasets, val_samplers):
            val_dataloaders.append(
                DataLoader(
                    dataset=dataset,
                    batch_size=self.hypers["batch_size"],
                    sampler=sampler,
                    shuffle=False,
                    drop_last=False,
                    collate_fn=collate_fn,
                )
            )
        val_dataloader = CombinedDataLoader(val_dataloaders, shuffle=False)

        train_targets = model.dataset_info.targets
        outputs_list = []
        for target_name, target_info in train_targets.items():
            outputs_list.append(target_name)
            for gradient_name in target_info.gradients:
                outputs_list.append(f"{target_name}_{gradient_name}_gradients")

        if self.hypers["USE_WEIGHT_DECAY"]:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.hypers["INITIAL_LR"],
                weight_decay=self.hypers["WEIGHT_DECAY"],
            )
        else:
            optimizer = torch.optim.Adam(
                model.parameters(), lr=self.hypers["INITIAL_LR"]
            )

        if self.optimizer_state_dict is not None:
            # try to load the optimizer state dict, but this is only possible
            # if there are no new targets in the model (new parameters)
            if not model.has_new_targets:
                optimizer.load_state_dict(self.optimizer_state_dict)

        lr_scheduler = get_scheduler(optimizer, self.hypers)

        if self.scheduler_state_dict is not None:
            # same as the optimizer, try to load the scheduler state dict
            if not model.has_new_targets:
                lr_scheduler.load_state_dict(self.scheduler_state_dict)

        per_structure_targets = self.hypers["per_structure_targets"]

        # Log the initial learning rate:
        old_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"Initial learning rate: {old_lr}")

        rotational_augmenter = RotationalAugmenter(train_targets)

        start_epoch = 0 if self.epoch is None else self.epoch + 1

        # Train the model:
        if self.best_metric is None:
            self.best_metric = float("inf")
        logger.info("Starting training")
        epoch = start_epoch

        history = []
        if MLIP_SETTINGS.USE_ENERGIES:
            energies_logger = FullLogger(
                FITTING_SCHEME.SUPPORT_MISSING_VALUES,
                FITTING_SCHEME.USE_SHIFT_AGNOSTIC_LOSS,
                device,
            )

        if MLIP_SETTINGS.USE_FORCES:
            forces_logger = FullLogger(
                FITTING_SCHEME.SUPPORT_MISSING_VALUES,
                FITTING_SCHEME.USE_SHIFT_AGNOSTIC_LOSS,
                device,
            )

        if MLIP_SETTINGS.USE_FORCES:
            val_forces = torch.cat(val_forces, dim=0)

            sliding_forces_rmse = get_rmse(
                val_forces.data.cpu().to(dtype=torch.float32).numpy(), 0.0
            )

            forces_rmse_model_keeper = ModelKeeper()
            forces_mae_model_keeper = ModelKeeper()

        if MLIP_SETTINGS.USE_ENERGIES:
            if FITTING_SCHEME.ENERGIES_LOSS == "per_structure":
                sliding_energies_rmse = get_rmse(val_energies, np.mean(val_energies))
            else:
                val_n_atoms = np.array(
                    [len(struc.positions) for struc in ase_val_dataset]
                )
                val_energies_per_atom = val_energies / val_n_atoms
                sliding_energies_rmse = get_rmse(
                    val_energies_per_atom, np.mean(val_energies_per_atom)
                )

            energies_rmse_model_keeper = ModelKeeper()
            energies_mae_model_keeper = ModelKeeper()

        if MLIP_SETTINGS.USE_ENERGIES and MLIP_SETTINGS.USE_FORCES:
            multiplication_rmse_model_keeper = ModelKeeper()
            multiplication_mae_model_keeper = ModelKeeper()

        if FITTING_SCHEME.EPOCHS_WARMUP > 0:
            remaining_lr_scheduler_steps = max(
                FITTING_SCHEME.EPOCHS_WARMUP - scheduler.last_epoch, 0
            )
            if remaining_lr_scheduler_steps > 0:
                lr_sheduler_msg = (
                    f" with {remaining_lr_scheduler_steps} steps of LR warmup"
                )
            else:
                lr_sheduler_msg = ""
        else:
            lr_sheduler_msg = ""
        logging.info(
            f"Starting training for {FITTING_SCHEME.EPOCH_NUM} epochs" + lr_sheduler_msg
        )
        TIME_TRAINING_STARTED = time.time()
        last_elapsed_time = 0
        if self.best_metric is None:
            self.best_metric = float("inf")
        start_epoch = 1 if self.epoch is None else self.epoch + 1
        for epoch in range(start_epoch, start_epoch + FITTING_SCHEME.EPOCH_NUM):
            pet_model.train(True)
            for batch in train_loader:
                if not FITTING_SCHEME.MULTI_GPU:
                    batch.to(device)

                if FITTING_SCHEME.MULTI_GPU:
                    pet_model.module.augmentation = True
                    pet_model.module.create_graph = True
                    predictions_energies, predictions_forces = pet_model(batch)
                else:
                    predictions_energies, predictions_forces = pet_model(
                        batch, augmentation=True, create_graph=True
                    )

                if FITTING_SCHEME.MULTI_GPU:
                    y_list = [el.y for el in batch]
                    batch_y = torch.tensor(
                        y_list, dtype=torch.get_default_dtype(), device=device
                    )

                    n_atoms_list = [el.n_atoms for el in batch]
                    batch_n_atoms = torch.tensor(
                        n_atoms_list, dtype=torch.get_default_dtype(), device=device
                    )

                else:
                    batch_y = batch.y
                    batch_n_atoms = batch.n_atoms

                if FITTING_SCHEME.ENERGIES_LOSS == "per_atom":
                    predictions_energies = predictions_energies / batch_n_atoms
                    ground_truth_energies = batch_y / batch_n_atoms
                else:
                    ground_truth_energies = batch_y

                if MLIP_SETTINGS.USE_ENERGIES:
                    energies_logger.train_logger.update(
                        predictions_energies, ground_truth_energies
                    )
                    loss_energies = get_loss(
                        predictions_energies,
                        ground_truth_energies,
                        FITTING_SCHEME.SUPPORT_MISSING_VALUES,
                        FITTING_SCHEME.USE_SHIFT_AGNOSTIC_LOSS,
                    )
                if MLIP_SETTINGS.USE_FORCES:
                    if FITTING_SCHEME.MULTI_GPU:
                        forces_list = [el.forces for el in batch]
                        batch_forces = torch.cat(forces_list, dim=0).to(device)
                    else:
                        batch_forces = batch.forces

                    forces_logger.train_logger.update(predictions_forces, batch_forces)
                    loss_forces = get_loss(
                        predictions_forces,
                        batch_forces,
                        FITTING_SCHEME.SUPPORT_MISSING_VALUES,
                        FITTING_SCHEME.USE_SHIFT_AGNOSTIC_LOSS,
                    )
                if MLIP_SETTINGS.USE_ENERGIES and MLIP_SETTINGS.USE_FORCES:
                    loss = FITTING_SCHEME.ENERGY_WEIGHT * loss_energies / (
                        sliding_energies_rmse**2
                    ) + loss_forces / (sliding_forces_rmse**2)
                    loss.backward()

                if MLIP_SETTINGS.USE_ENERGIES and (not MLIP_SETTINGS.USE_FORCES):
                    loss_energies.backward()
                if MLIP_SETTINGS.USE_FORCES and (not MLIP_SETTINGS.USE_ENERGIES):
                    loss_forces.backward()

                if FITTING_SCHEME.DO_GRADIENT_CLIPPING:
                    torch.nn.utils.clip_grad_norm_(
                        pet_model.parameters(),
                        max_norm=FITTING_SCHEME.GRADIENT_CLIPPING_MAX_NORM,
                    )
                optim.step()
                optim.zero_grad()

            pet_model.train(False)
            for batch in val_loader:
                if not FITTING_SCHEME.MULTI_GPU:
                    batch.to(device)

                if FITTING_SCHEME.MULTI_GPU:
                    pet_model.module.augmentation = False
                    pet_model.module.create_graph = False
                    predictions_energies, predictions_forces = pet_model(batch)
                else:
                    predictions_energies, predictions_forces = pet_model(
                        batch, augmentation=False, create_graph=False
                    )

                if FITTING_SCHEME.MULTI_GPU:
                    y_list = [el.y for el in batch]
                    batch_y = torch.tensor(
                        y_list, dtype=torch.get_default_dtype(), device=device
                    )

                    n_atoms_list = [el.n_atoms for el in batch]
                    batch_n_atoms = torch.tensor(
                        n_atoms_list, dtype=torch.get_default_dtype(), device=device
                    )
                else:
                    batch_y = batch.y
                    batch_n_atoms = batch.n_atoms

                if FITTING_SCHEME.ENERGIES_LOSS == "per_atom":
                    predictions_energies = predictions_energies / batch_n_atoms
                    ground_truth_energies = batch_y / batch_n_atoms
                else:
                    ground_truth_energies = batch_y

                if MLIP_SETTINGS.USE_ENERGIES:
                    energies_logger.val_logger.update(
                        predictions_energies, ground_truth_energies
                    )
                if MLIP_SETTINGS.USE_FORCES:
                    if FITTING_SCHEME.MULTI_GPU:
                        forces_list = [el.forces for el in batch]
                        batch_forces = torch.cat(forces_list, dim=0).to(device)
                    else:
                        batch_forces = batch.forces
                    forces_logger.val_logger.update(predictions_forces, batch_forces)

            now = {}
            if FITTING_SCHEME.ENERGIES_LOSS == "per_structure":
                energies_key = "energies per structure"
            else:
                energies_key = "energies per atom"

            if MLIP_SETTINGS.USE_ENERGIES:
                now[energies_key] = energies_logger.flush()

            if MLIP_SETTINGS.USE_FORCES:
                now["forces"] = forces_logger.flush()
            now["lr"] = scheduler.get_last_lr()
            now["epoch"] = epoch

            now["elapsed_time"] = time.time() - TIME_TRAINING_STARTED
            now["epoch_time"] = now["elapsed_time"] - last_elapsed_time
            now["estimated_remaining_time"] = (now["elapsed_time"] / epoch) * (
                FITTING_SCHEME.EPOCH_NUM - epoch
            )
            last_elapsed_time = now["elapsed_time"]

            if MLIP_SETTINGS.USE_ENERGIES:
                sliding_energies_rmse = (
                    FITTING_SCHEME.SLIDING_FACTOR * sliding_energies_rmse
                    + (1.0 - FITTING_SCHEME.SLIDING_FACTOR)
                    * now[energies_key]["val"]["rmse"]
                )

                energies_mae_model_keeper.update(
                    pet_model, now[energies_key]["val"]["mae"], epoch
                )
                energies_rmse_model_keeper.update(
                    pet_model, now[energies_key]["val"]["rmse"], epoch
                )

            if MLIP_SETTINGS.USE_FORCES:
                sliding_forces_rmse = (
                    FITTING_SCHEME.SLIDING_FACTOR * sliding_forces_rmse
                    + (1.0 - FITTING_SCHEME.SLIDING_FACTOR)
                    * now["forces"]["val"]["rmse"]
                )
                forces_mae_model_keeper.update(
                    pet_model, now["forces"]["val"]["mae"], epoch
                )
                forces_rmse_model_keeper.update(
                    pet_model, now["forces"]["val"]["rmse"], epoch
                )

            if MLIP_SETTINGS.USE_ENERGIES and MLIP_SETTINGS.USE_FORCES:
                multiplication_mae_model_keeper.update(
                    pet_model,
                    now["forces"]["val"]["mae"] * now[energies_key]["val"]["mae"],
                    epoch,
                    additional_info=[
                        now[energies_key]["val"]["mae"],
                        now["forces"]["val"]["mae"],
                    ],
                )
                multiplication_rmse_model_keeper.update(
                    pet_model,
                    now["forces"]["val"]["rmse"] * now[energies_key]["val"]["rmse"],
                    epoch,
                    additional_info=[
                        now[energies_key]["val"]["rmse"],
                        now["forces"]["val"]["rmse"],
                    ],
                )
            last_lr = scheduler.get_last_lr()[0]
            log_epoch_stats(epoch, FITTING_SCHEME.EPOCH_NUM, now, last_lr, energies_key)

            history.append(now)
            scheduler.step()
            elapsed = time.time() - TIME_SCRIPT_STARTED
            if epoch > 0 and epoch % FITTING_SCHEME.CHECKPOINT_INTERVAL == 0:
                self.epoch = epoch
                pet_checkpoint = {
                    "model_state_dict": pet_model.state_dict(),
                    "optim_state_dict": optim.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "dtype_used": dtype2string(dtype),
                }
                torch.save(
                    pet_checkpoint,
                    f"{checkpoint_dir}/{NAME_OF_CALCULATION}/checkpoint_{epoch}",
                )
                trainer_state_dict = {
                    "optim_state_dict": pet_checkpoint["optim_state_dict"],
                    "scheduler_state_dict": pet_checkpoint["scheduler_state_dict"],
                }
                last_model_state_dict = pet_checkpoint["model_state_dict"]
                if model.is_lora_applied:
                    lora_state_dict = {
                        "lora_rank": FITTING_SCHEME.LORA_RANK,
                        "lora_alpha": FITTING_SCHEME.LORA_ALPHA,
                    }
                else:
                    lora_state_dict = None
                last_model_checkpoint = {
                    "architecture_name": "pet",
                    "trainer_state_dict": trainer_state_dict,
                    "model_state_dict": last_model_state_dict,
                    "best_model_state_dict": self.best_model_state_dict,
                    "best_metric": self.best_metric,
                    "hypers": self.hypers,
                    "epoch": self.epoch,
                    "dataset_info": model.dataset_info,
                    "self_contributions": self_contributions,
                    "lora_state_dict": lora_state_dict,
                }
                torch.save(
                    last_model_checkpoint,
                    f"{checkpoint_dir}/model.ckpt_{epoch}",
                )

            if FITTING_SCHEME.MAX_TIME is not None:
                if elapsed > FITTING_SCHEME.MAX_TIME:
                    logging.info("Reached maximum time\n")
                    break
        logging.info("Training is finished\n")
        logging.info("Saving the model and history...")
        torch.save(
            {
                "model_state_dict": pet_model.state_dict(),
                "optim_state_dict": optim.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "dtype_used": dtype2string(dtype),
            },
            f"{checkpoint_dir}/{NAME_OF_CALCULATION}/checkpoint",
        )
        with open(f"{checkpoint_dir}/{NAME_OF_CALCULATION}/history.pickle", "wb") as f:
            pickle.dump(history, f)

        def save_model(model_name, model_keeper):
            torch.save(
                model_keeper.best_model.state_dict(),
                f"{checkpoint_dir}/{NAME_OF_CALCULATION}/{model_name}_state_dict",
            )

        summary = ""
        if MLIP_SETTINGS.USE_ENERGIES:
            if FITTING_SCHEME.ENERGIES_LOSS == "per_structure":
                postfix = "per structure"
            if FITTING_SCHEME.ENERGIES_LOSS == "per_atom":
                postfix = "per atom"
            save_model("best_val_mae_energies_model", energies_mae_model_keeper)
            summary += f"best val mae in energies {postfix}: "
            summary += f"{energies_mae_model_keeper.best_error} "
            summary += f"at epoch {energies_mae_model_keeper.best_epoch}\n"

            save_model("best_val_rmse_energies_model", energies_rmse_model_keeper)
            summary += f"best val rmse in energies {postfix}: "
            summary += f"{energies_rmse_model_keeper.best_error} "
            summary += f"at epoch {energies_rmse_model_keeper.best_epoch}\n"

            if energies_mae_model_keeper.best_error < self.best_metric:
                self.best_metric = energies_mae_model_keeper.best_error
                self.best_model_state_dict = (
                    energies_mae_model_keeper.best_model.state_dict()
                )

        if MLIP_SETTINGS.USE_FORCES:
            save_model("best_val_mae_forces_model", forces_mae_model_keeper)
            summary += f"best val mae in forces: {forces_mae_model_keeper.best_error} "
            summary += f"at epoch {forces_mae_model_keeper.best_epoch}\n"

            save_model("best_val_rmse_forces_model", forces_rmse_model_keeper)
            summary += (
                f"best val rmse in forces: {forces_rmse_model_keeper.best_error} "
            )
            summary += f"at epoch {forces_rmse_model_keeper.best_epoch}\n"

            if forces_mae_model_keeper.best_error < self.best_metric:
                self.best_metric = forces_mae_model_keeper.best_error
                self.best_model_state_dict = (
                    forces_mae_model_keeper.best_model.state_dict()
                )

        if MLIP_SETTINGS.USE_ENERGIES and MLIP_SETTINGS.USE_FORCES:
            save_model("best_val_mae_both_model", multiplication_mae_model_keeper)
            summary += f"best both (multiplication) mae in energies {postfix}: "
            summary += (
                f"{multiplication_mae_model_keeper.additional_info[0]} in forces: "
            )
            summary += f"{multiplication_mae_model_keeper.additional_info[1]} "
            summary += f"at epoch {multiplication_mae_model_keeper.best_epoch}\n"

            save_model("best_val_rmse_both_model", multiplication_rmse_model_keeper)
            summary += f"best both (multiplication) rmse in energies {postfix}: "
            summary += (
                f"{multiplication_rmse_model_keeper.additional_info[0]} in forces: "
            )
            summary += (
                f"{multiplication_rmse_model_keeper.additional_info[1]} at epoch "
            )
            summary += f"{multiplication_rmse_model_keeper.best_epoch}\n"

            if multiplication_mae_model_keeper.best_error < self.best_metric:
                self.best_metric = multiplication_mae_model_keeper.best_error
                self.best_model_state_dict = (
                    multiplication_mae_model_keeper.best_model.state_dict()
                )

        with open(f"{checkpoint_dir}/{NAME_OF_CALCULATION}/summary.txt", "wb") as f:
            f.write(summary.encode())
        logging.info(f"Total elapsed time: {time.time() - TIME_SCRIPT_STARTED}")

        ##########################################
        # FINISHING THE PURE PET TRAINING SCRIPT #
        ##########################################
        self.epoch = epoch
        wrapper = load_raw_pet_model(
            self.best_model_state_dict,
            model.hypers,
            all_species,
            self_contributions,
            use_lora_peft=FITTING_SCHEME.USE_LORA_PEFT,
            lora_rank=FITTING_SCHEME.LORA_RANK,
            lora_alpha=FITTING_SCHEME.LORA_ALPHA,
        )
        model.set_trained_model(wrapper)

    def save_checkpoint(self, model, path: Union[str, Path]):
        # This function takes a checkpoint from the PET folder and saves it
        # together with the hypers inside a file that will act as a metatrain
        # checkpoint
        pet_checkpoint = torch.load(
            self.pet_dir / "checkpoint", weights_only=False, map_location="cpu"
        )
        trainer_state_dict = {
            "optim_state_dict": pet_checkpoint["optim_state_dict"],
            "scheduler_state_dict": pet_checkpoint["scheduler_state_dict"],
        }
        last_model_state_dict = pet_checkpoint["model_state_dict"]
        if model.is_lora_applied:
            lora_state_dict = {
                "lora_rank": model.pet.model.rank,
                "lora_alpha": model.pet.model.alpha,
            }
        else:
            lora_state_dict = None
        last_model_checkpoint = {
            "architecture_name": "pet",
            "trainer_state_dict": trainer_state_dict,
            "model_state_dict": last_model_state_dict,
            "best_model_state_dict": self.best_model_state_dict,
            "best_metric": self.best_metric,
            "hypers": self.hypers,
            "epoch": self.epoch,
            "dataset_info": model.dataset_info,
            "self_contributions": model.pet.self_contributions.numpy(),
            "lora_state_dict": lora_state_dict,
        }
        best_model_checkpoint = {
            "architecture_name": "pet",
            "trainer_state_dict": None,
            "model_state_dict": self.best_model_state_dict,
            "best_model_state_dict": None,
            "best_metric": None,
            "hypers": self.hypers,
            "epoch": None,
            "dataset_info": model.dataset_info,
            "self_contributions": model.pet.self_contributions.numpy(),
            "lora_state_dict": lora_state_dict,
        }

        torch.save(
            last_model_checkpoint,
            check_file_extension(f"last_checkpoint_{str(path)}", ".ckpt"),
        )

        torch.save(
            best_model_checkpoint, check_file_extension("best_" + str(path), ".ckpt")
        )

    @classmethod
    def load_checkpoint(cls, path: Union[str, Path], train_hypers) -> "Trainer":
        # This function loads a metatrain PET checkpoint and returns a Trainer
        # instance with the hypers, while also saving the checkpoint in the
        # class
        checkpoint = torch.load(path, weights_only=False, map_location="cpu")
        trainer = cls(train_hypers)
        trainer.pet_trainer_state = checkpoint["trainer_state_dict"]
        trainer.epoch = checkpoint["epoch"]
        old_fitting_scheme = checkpoint["hypers"]["FITTING_SCHEME"]
        new_fitting_scheme = train_hypers
        best_metric = checkpoint["best_metric"]
        best_model_state_dict = checkpoint["best_model_state_dict"]
        # The following code is not reached in the current implementation
        # because the check for the train targets is done earlier in the
        # training process, and changing the training targets between the
        # runs is forbidden. However, this code is kept here for future reference.
        for key in new_fitting_scheme:
            if key in ["USE_ENERGIES", "USE_FORCES"]:
                if new_fitting_scheme[key] != old_fitting_scheme[key]:
                    logger.warning(
                        f"The {key} training hyperparameter was changed from "
                        f"{old_fitting_scheme[key]} to {new_fitting_scheme[key]} "
                        "inbetween the last checkpoint and the current training. "
                        "The `best model` and the `best loss` parts of the checkpoint "
                        "will be reset to avoid inconsistencies."
                    )
                    best_metric = None
                    best_model_state_dict = None
        trainer.best_metric = best_metric
        trainer.best_model_state_dict = best_model_state_dict
        return trainer
