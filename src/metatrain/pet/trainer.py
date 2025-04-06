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

from ..utils.data import Dataset, check_datasets
from ..utils.io import check_file_extension
from . import PET as WrappedPET
from .modules.analysis import adapt_hypers
from .modules.data_preparation import (
    get_corrected_energies,
    get_forces,
    get_pyg_graphs,
    get_self_contributions,
    update_pyg_graphs,
)
from .modules.hypers import Hypers, save_hypers
from .modules.pet import PET, FlagsWrapper, PETMLIPWrapper, PETUtilityWrapper
from .modules.utilities import (
    FullLogger,
    ModelKeeper,
    dtype2string,
    get_calc_names,
    get_data_loaders,
    get_loss,
    get_optimizer,
    get_rmse,
    get_scheduler,
    log_epoch_stats,
    set_reproducibility,
    string2dtype,
)
from .utils import dataset_to_ase, load_raw_pet_model, update_hypers
from .utils.fine_tuning import LoRAWrapper


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
        model: WrappedPET,
        dtype: torch.dtype,
        devices: List[torch.device],
        train_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        val_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        checkpoint_dir: str,
    ):
        assert dtype in WrappedPET.__supported_dtypes__

        name_of_calculation = "pet"
        self.pet_dir = Path(checkpoint_dir) / name_of_calculation

        if len(train_datasets) != 1:
            raise ValueError("PET only supports a single training dataset")
        if len(val_datasets) != 1:
            raise ValueError("PET only supports a single validation dataset")
        if torch.device("cpu") in devices:
            warnings.warn(
                "Training PET on a CPU is very slow! For better performance, use a "
                "CUDA GPU.",
                stacklevel=1,
            )

        logging.info("Checking datasets for consistency")
        check_datasets(train_datasets, val_datasets)

        train_dataset = train_datasets[0]
        val_dataset = val_datasets[0]

        # are we fitting on only energies or energies and forces?
        target_name = model.target_name
        do_forces = (
            next(iter(train_dataset))[target_name].block().has_gradient("positions")
        )

        ase_train_dataset = dataset_to_ase(
            train_dataset, model, do_forces=do_forces, target_name=target_name
        )
        ase_val_dataset = dataset_to_ase(
            val_dataset, model, do_forces=do_forces, target_name=target_name
        )

        self.hypers = update_hypers(self.hypers, model.hypers, do_forces)

        device = devices[0]  # only one device, as we don't support multi-gpu for now

        ########################################
        # STARTING THE PURE PET TRAINING SCRIPT #
        ########################################

        logging.info("Initializing PET training...")

        TIME_SCRIPT_STARTED = time.time()
        value = datetime.datetime.fromtimestamp(TIME_SCRIPT_STARTED)
        logging.info(f"Starting training at: {value.strftime('%Y-%m-%d %H:%M:%S')}")
        training_configuration_log = "Training configuration:\n"
        training_configuration_log += f"Output directory: {checkpoint_dir}\n"
        training_configuration_log += f"Training using device: {device}\n"

        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)

        hypers = Hypers(self.hypers)
        dtype = string2dtype(hypers.ARCHITECTURAL_HYPERS.DTYPE)  # type: ignore
        torch.set_default_dtype(dtype)

        FITTING_SCHEME = hypers.FITTING_SCHEME  # type: ignore
        MLIP_SETTINGS = hypers.MLIP_SETTINGS  # type: ignore
        ARCHITECTURAL_HYPERS = hypers.ARCHITECTURAL_HYPERS  # type: ignore

        if model.is_lora_applied and not FITTING_SCHEME.USE_LORA_PEFT:
            raise ValueError(
                "LoRA is applied to the model, but the USE_LORA_PEFT is False"
                " in the training hyperparameters. Please set USE_LORA_PEFT to True"
            )

        if FITTING_SCHEME.USE_SHIFT_AGNOSTIC_LOSS:
            raise ValueError(
                "shift agnostic loss is intended only for general target training"
            )

        training_configuration_log += (
            f"Output dimensionality: {ARCHITECTURAL_HYPERS.D_OUTPUT}\n"
        )
        training_configuration_log += (
            f"Target type: {ARCHITECTURAL_HYPERS.TARGET_TYPE}\n"
        )
        training_configuration_log += (
            f"Target aggregation: {ARCHITECTURAL_HYPERS.TARGET_AGGREGATION}\n"
        )
        training_configuration_log += f"Random seed: {FITTING_SCHEME.RANDOM_SEED}\n"
        training_configuration_log += (
            f"CUDA is deterministic: {FITTING_SCHEME.CUDA_DETERMINISTIC}"
        )

        st = """
Legend: LR          -> Learning Rate
        MAE         -> Mean Absolute Error
        RMSE        -> Root Mean Square Error
        V-E-MAE/at  -> MAE of the Energy per atom on the Validation set
        V-E-RMSE/at -> RMSE of the Energy per atom on the Validation set
        V-F-MAE     -> MAE of the Forces on the Validation set
        V-F-RMSE    -> RMSE of the Forces on the Validation set
        T-E-MAE/at  -> MAE of the Energy per atom on the Training set
        T-E-RMSE/at -> RMSE of the Energy per atom on the Training set
        T-F-MAE     -> MAE of the Forces on the Training set
        T-F-RMSE    -> RMSE of the Forces on the Training set
Units of the Energy and Forces are the same units given in input"""
        training_configuration_log += st
        logging.info(training_configuration_log)

        set_reproducibility(
            FITTING_SCHEME.RANDOM_SEED, FITTING_SCHEME.CUDA_DETERMINISTIC
        )

        adapt_hypers(FITTING_SCHEME, ase_train_dataset)

        all_species = model.atomic_types

        name_to_load, NAME_OF_CALCULATION = get_calc_names(
            os.listdir(checkpoint_dir), name_of_calculation
        )

        os.mkdir(f"{checkpoint_dir}/{NAME_OF_CALCULATION}")
        np.save(f"{checkpoint_dir}/{NAME_OF_CALCULATION}/all_species.npy", all_species)
        hypers.UTILITY_FLAGS.CALCULATION_TYPE = "mlip"  # type: ignore
        save_hypers(hypers, f"{checkpoint_dir}/{NAME_OF_CALCULATION}/hypers_used.yaml")

        logging.info("Convering structures to PyG graphs...")

        train_graphs = get_pyg_graphs(
            ase_train_dataset,
            all_species,
            ARCHITECTURAL_HYPERS.R_CUT,
            ARCHITECTURAL_HYPERS.USE_ADDITIONAL_SCALAR_ATTRIBUTES,
            ARCHITECTURAL_HYPERS.USE_LONG_RANGE,
            ARCHITECTURAL_HYPERS.K_CUT,
            ARCHITECTURAL_HYPERS.N_TARGETS > 1,
            ARCHITECTURAL_HYPERS.TARGET_INDEX_KEY,
        )
        val_graphs = get_pyg_graphs(
            ase_val_dataset,
            all_species,
            ARCHITECTURAL_HYPERS.R_CUT,
            ARCHITECTURAL_HYPERS.USE_ADDITIONAL_SCALAR_ATTRIBUTES,
            ARCHITECTURAL_HYPERS.USE_LONG_RANGE,
            ARCHITECTURAL_HYPERS.K_CUT,
            ARCHITECTURAL_HYPERS.N_TARGETS > 1,
            ARCHITECTURAL_HYPERS.TARGET_INDEX_KEY,
        )

        logging.info("Pre-processing training data...")
        if MLIP_SETTINGS.USE_ENERGIES:
            if model.pet is not None:
                self_contributions = model.pet.self_contributions
            else:
                self_contributions = get_self_contributions(
                    MLIP_SETTINGS.ENERGY_KEY, ase_train_dataset, all_species
                )
            np.save(
                f"{checkpoint_dir}/{NAME_OF_CALCULATION}/self_contributions.npy",
                self_contributions,
            )

            train_energies = get_corrected_energies(
                MLIP_SETTINGS.ENERGY_KEY,
                ase_train_dataset,
                all_species,
                self_contributions,
            )
            val_energies = get_corrected_energies(
                MLIP_SETTINGS.ENERGY_KEY,
                ase_val_dataset,
                all_species,
                self_contributions,
            )

            update_pyg_graphs(train_graphs, "y", train_energies)
            update_pyg_graphs(val_graphs, "y", val_energies)

        if MLIP_SETTINGS.USE_FORCES:
            train_forces = get_forces(ase_train_dataset, MLIP_SETTINGS.FORCES_KEY)
            val_forces = get_forces(ase_val_dataset, MLIP_SETTINGS.FORCES_KEY)

            update_pyg_graphs(train_graphs, "forces", train_forces)
            update_pyg_graphs(val_graphs, "forces", val_forces)

        train_loader, val_loader = get_data_loaders(
            train_graphs, val_graphs, FITTING_SCHEME
        )

        logging.info("Initializing the model...")
        if model.pet is not None:
            pet_model = model.pet.model
            if model.is_lora_applied:
                pet_model.model.hypers.TARGET_TYPE = "structural"
                pet_model.model.TARGET_TYPE = "structural"
            else:
                pet_model.hypers.TARGET_TYPE = "structural"
                pet_model.TARGET_TYPE = "structural"
        else:
            pet_model = PET(ARCHITECTURAL_HYPERS, 0.0, len(all_species))
        num_params = sum([p.numel() for p in pet_model.parameters()])
        logging.info(f"Number of parameters: {num_params}")

        if FITTING_SCHEME.USE_LORA_PEFT:
            if not model.is_lora_applied:
                lora_rank = FITTING_SCHEME.LORA_RANK
                lora_alpha = FITTING_SCHEME.LORA_ALPHA
                pet_model = LoRAWrapper(pet_model, lora_rank, lora_alpha)
                model.is_lora_applied = True

            num_trainable_params = sum(
                [p.numel() for p in pet_model.parameters() if p.requires_grad]
            )
            fraction = num_trainable_params / num_params * 100
            logging.info(
                f"Using LoRA PEFT with rank {FITTING_SCHEME.LORA_RANK} "
                + f"and alpha {FITTING_SCHEME.LORA_ALPHA}"
            )
            logging.info(
                "Number of trainable parameters: "
                + f"{num_trainable_params} [{fraction:.2f}%]"
            )
        pet_model = pet_model.to(device=device, dtype=dtype)
        pet_model = PETUtilityWrapper(pet_model, FITTING_SCHEME.GLOBAL_AUG)

        pet_model = PETMLIPWrapper(
            pet_model, MLIP_SETTINGS.USE_ENERGIES, MLIP_SETTINGS.USE_FORCES
        )
        if FITTING_SCHEME.MULTI_GPU and torch.cuda.is_available():
            logging.info(
                f"Using multi-GPU training on {torch.cuda.device_count()} GPUs"
            )
            pet_model = DataParallel(FlagsWrapper(pet_model))
            pet_model = pet_model.to(torch.device("cuda:0"))

        optim = get_optimizer(pet_model, FITTING_SCHEME)
        scheduler = get_scheduler(optim, FITTING_SCHEME)

        if self.pet_trainer_state is not None:
            for i, param_group in enumerate(optim.param_groups):
                if len(param_group["params"]) != len(
                    self.pet_trainer_state["optim_state_dict"]["param_groups"][i][
                        "params"
                    ]
                ):
                    raise RuntimeError(
                        "The number of parameters in the optimizer state dict "
                        "from the loaded checkpoint does not match the current "
                        "optimizer state. This means the model architecture has "
                        "changed since the last checkpoint. If you are using LoRA "
                        "PEFT, you should use the best model chekpoint from the "
                        "pre-training step. If you still need to use the current "
                        "checkpoint, set the trainer_state_dict in the checkpoint "
                        "to None and restart the training."
                    )

            optim.load_state_dict(self.pet_trainer_state["optim_state_dict"])
            scheduler.load_state_dict(self.pet_trainer_state["scheduler_state_dict"])
        else:
            logging.info(
                "No optimizer and scheduler state found in the "
                "checkpoint, starting from scratch"
            )

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
                    logging.warning(
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
