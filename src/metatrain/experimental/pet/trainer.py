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
from pet.analysis import adapt_hypers
from pet.data_preparation import (
    get_all_species,
    get_corrected_energies,
    get_forces,
    get_pyg_graphs,
    get_self_contributions,
    update_pyg_graphs,
)
from pet.hypers import Hypers, save_hypers
from pet.pet import (
    PET,
    FlagsWrapper,
    PETMLIPWrapper,
    PETUtilityWrapper,
    SelfContributionsWrapper,
)
from pet.utilities import (
    FullLogger,
    ModelKeeper,
    dtype2string,
    get_calc_names,
    get_data_loaders,
    get_loss,
    get_optimizer,
    get_rmse,
    get_scheduler,
    load_checkpoint,
    log_epoch_stats,
    set_reproducibility,
    string2dtype,
)
from torch_geometric.nn import DataParallel

from ...utils.data import Dataset, check_datasets
from . import PET as WrappedPET
from .utils import dataset_to_ase, update_hypers, update_state_dict
from .utils.fine_tuning import LoRAWrapper


logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, train_hypers):
        self.hypers = {"FITTING_SCHEME": train_hypers}
        self.pet_dir = None
        self.pet_checkpoint = None

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

        logger.info("Checking datasets for consistency")
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

        if self.pet_checkpoint is not None:
            # save the checkpoint to a temporary file, so that fit_pet can load it
            checkpoint_path = Path(checkpoint_dir) / "checkpoint.temp"
            torch.save(
                self.pet_checkpoint,
                checkpoint_path,
            )
        else:
            checkpoint_path = None

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
        dtype = string2dtype(hypers.ARCHITECTURAL_HYPERS.DTYPE)
        torch.set_default_dtype(dtype)

        FITTING_SCHEME = hypers.FITTING_SCHEME
        MLIP_SETTINGS = hypers.MLIP_SETTINGS
        ARCHITECTURAL_HYPERS = hypers.ARCHITECTURAL_HYPERS

        if FITTING_SCHEME.USE_SHIFT_AGNOSTIC_LOSS:
            raise ValueError(
                "shift agnostic loss is intended only for general target training"
            )

        ARCHITECTURAL_HYPERS.D_OUTPUT = 1  # energy is a single scalar
        ARCHITECTURAL_HYPERS.TARGET_TYPE = "structural"  # energy is structural property
        ARCHITECTURAL_HYPERS.TARGET_AGGREGATION = (
            "sum"  # energy is a sum of atomic energies
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
Legend: LR       -> Learning Rate
        MAE      -> Mean Square Error
        RMSE     -> Root Mean Square Error
        V-E-MAE  -> MAE of the Energy on the validation set
        V-E-RMSE -> RMSE of the Energy on the validation set
        V-F-MAE  -> MAE of the Forces on the validation set
        V-F-RMSE -> RMSE of the Forces on the validation set
        T-E-MAE  -> MAE of the Energy on the training set
        T-E-RMSE -> RMSE of the Energy on the training set
        T-F-MAE  -> MAE of the Forces on the training set
        T-F-RMSE -> RMSE of the Forces on the training set"""
        training_configuration_log += st
        logging.info(training_configuration_log)

        set_reproducibility(
            FITTING_SCHEME.RANDOM_SEED, FITTING_SCHEME.CUDA_DETERMINISTIC
        )

        adapt_hypers(FITTING_SCHEME, ase_train_dataset)
        dataset = ase_train_dataset + ase_val_dataset

        all_dataset_species = get_all_species(dataset)

        if FITTING_SCHEME.ALL_SPECIES_PATH is not None:
            logging.info(f"Loading all species from: {FITTING_SCHEME.ALL_SPECIES_PATH}")
            all_species = np.load(FITTING_SCHEME.ALL_SPECIES_PATH)
            if not np.all(np.isin(all_dataset_species, all_species)):
                raise ValueError(
                    "For the model fine-tuning, the set of species in the dataset "
                    "must be a subset of the set of species in the pre-trained model. "
                    "Please check, if the ALL_SPECIES_PATH is file contains all the "
                    "elements from the fine-tuning dataset."
                )
            model.atomic_types = all_species.tolist()
            model.dataset_info.atomic_types = all_species.tolist()
        else:
            all_species = all_dataset_species

        name_to_load, NAME_OF_CALCULATION = get_calc_names(
            os.listdir(checkpoint_dir), name_of_calculation
        )

        os.mkdir(f"{checkpoint_dir}/{NAME_OF_CALCULATION}")
        np.save(f"{checkpoint_dir}/{NAME_OF_CALCULATION}/all_species.npy", all_species)
        hypers.UTILITY_FLAGS.CALCULATION_TYPE = "mlip"
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
            if FITTING_SCHEME.SELF_CONTRIBUTIONS_PATH is not None:
                self_contributions_path = FITTING_SCHEME.SELF_CONTRIBUTIONS_PATH
                logging.info(
                    f"Loading self contributions from: {self_contributions_path}"
                )
                self_contributions = np.load(FITTING_SCHEME.SELF_CONTRIBUTIONS_PATH)
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
        pet_model = PET(ARCHITECTURAL_HYPERS, 0.0, len(all_species)).to(device)
        num_params = sum([p.numel() for p in pet_model.parameters()])
        logging.info(f"Number of parameters: {num_params}")

        if FITTING_SCHEME.MODEL_TO_START_WITH is not None:
            logging.info(f"Loading model from: {FITTING_SCHEME.MODEL_TO_START_WITH}")
            state_dict = torch.load(
                FITTING_SCHEME.MODEL_TO_START_WITH, weights_only=True
            )
            new_state_dict = update_state_dict(state_dict)
            loaded_keys = set(new_state_dict.keys())
            current_keys = set(pet_model.state_dict().keys())
            if not loaded_keys == current_keys:
                raise ValueError(
                    "The keys of the loaded model do not match the keys of the "
                    "current model. Please check, if the current model hypers are "
                    "compatible with the loaded model."
                )
            pet_model.load_state_dict(new_state_dict)
            pet_model = pet_model.to(dtype=dtype)
            if ARCHITECTURAL_HYPERS.USE_LORA_PEFT:
                for param in pet_model.parameters():
                    param.requires_grad = False
                lora_rank = ARCHITECTURAL_HYPERS.LORA_RANK
                lora_alpha = ARCHITECTURAL_HYPERS.LORA_ALPHA
                pet_model = LoRAWrapper(pet_model, lora_rank, lora_alpha).to(device)
                num_trainable_params = sum(
                    [p.numel() for p in pet_model.parameters() if p.requires_grad]
                )
                fraction = num_trainable_params / num_params * 100
                logging.info(
                    f"Using LoRA PEFT with rank {lora_rank} and alpha {lora_alpha}"
                )
                logging.info(
                    "Number of trainable parameters: "
                    + f"{num_trainable_params} [{fraction:.2f}%]"
                )

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

        if checkpoint_path is not None:
            logging.info(f"Loading model and checkpoint from: {checkpoint_path}\n")
            load_checkpoint(pet_model, optim, scheduler, checkpoint_path)
        elif name_to_load is not None:
            path = f"{checkpoint_dir}/{name_to_load}/checkpoint"
            logging.info(f"Loading model and checkpoint from: {path}\n")
            load_checkpoint(
                pet_model,
                optim,
                scheduler,
                f"{checkpoint_dir}/{name_to_load}/checkpoint",
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
        for epoch in range(1, FITTING_SCHEME.EPOCH_NUM + 1):
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
                    # print('batch_y: ', batch_y.shape)
                    # print('batch_n_atoms: ', batch_n_atoms.shape)

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

                    # print('batch_y: ', batch_y.shape)
                    # print('batch_n_atoms: ', batch_n_atoms.shape)
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
                checkpoint_dict = {
                    "model_state_dict": pet_model.state_dict(),
                    "optim_state_dict": optim.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "dtype_used": dtype2string(dtype),
                }
                torch.save(
                    checkpoint_dict,
                    f"{checkpoint_dir}/{NAME_OF_CALCULATION}/checkpoint_{epoch}",
                )
                torch.save(
                    {
                        "checkpoint": checkpoint_dict,
                        "hypers": self.hypers,
                        "dataset_info": model.dataset_info,
                        "self_contributions": np.load(
                            self.pet_dir / "self_contributions.npy"  # type: ignore
                        ),
                    },
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

        if MLIP_SETTINGS.USE_FORCES:
            save_model("best_val_mae_forces_model", forces_mae_model_keeper)
            summary += f"best val mae in forces: {forces_mae_model_keeper.best_error} "
            summary += f"at epoch {forces_mae_model_keeper.best_epoch}\n"

            save_model("best_val_rmse_forces_model", forces_rmse_model_keeper)
            summary += (
                f"best val rmse in forces: {forces_rmse_model_keeper.best_error} "
            )
            summary += f"at epoch {forces_rmse_model_keeper.best_epoch}\n"

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

        with open(f"{checkpoint_dir}/{NAME_OF_CALCULATION}/summary.txt", "wb") as f:
            f.write(summary.encode())
        logging.info(f"Total elapsed time: {time.time() - TIME_SCRIPT_STARTED}")

        ##########################################
        # FINISHING THE PURE PET TRAINING SCRIPT #
        ##########################################

        if self.pet_checkpoint is not None:
            # remove the temporary file
            os.remove(Path(checkpoint_dir) / "checkpoint.temp")

        if do_forces:
            load_path = self.pet_dir / "best_val_mae_forces_model_state_dict"
        else:
            load_path = self.pet_dir / "best_val_mae_both_model_state_dict"

        state_dict = torch.load(load_path, weights_only=True)
        ARCHITECTURAL_HYPERS = Hypers(model.hypers)
        raw_pet = PET(ARCHITECTURAL_HYPERS, 0.0, len(all_species))
        if ARCHITECTURAL_HYPERS.USE_LORA_PEFT:
            lora_rank = ARCHITECTURAL_HYPERS.LORA_RANK
            lora_alpha = ARCHITECTURAL_HYPERS.LORA_ALPHA
            raw_pet = LoRAWrapper(raw_pet, lora_rank, lora_alpha)

        new_state_dict = update_state_dict(state_dict)
        raw_pet.load_state_dict(new_state_dict)

        self_contributions_path = self.pet_dir / "self_contributions.npy"
        self_contributions = np.load(self_contributions_path)
        wrapper = SelfContributionsWrapper(raw_pet, self_contributions)

        model.set_trained_model(wrapper)

    def save_checkpoint(self, model, path: Union[str, Path]):
        # This function takes a checkpoint from the PET folder and saves it
        # together with the hypers inside a file that will act as a metatrain
        # checkpoint
        checkpoint_path = self.pet_dir / "checkpoint"  # type: ignore
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
        torch.save(
            {
                "checkpoint": checkpoint,
                "hypers": self.hypers,
                "dataset_info": model.dataset_info,
                "self_contributions": np.load(
                    self.pet_dir / "self_contributions.npy"  # type: ignore
                ),
            },
            path,
        )

    @classmethod
    def load_checkpoint(cls, path: Union[str, Path], train_hypers) -> "Trainer":
        # This function loads a metatrain PET checkpoint and returns a Trainer
        # instance with the hypers, while also saving the checkpoint in the
        # class
        checkpoint = torch.load(path, weights_only=False, map_location="cpu")
        trainer = cls(train_hypers)
        trainer.pet_checkpoint = checkpoint["checkpoint"]
        return trainer
