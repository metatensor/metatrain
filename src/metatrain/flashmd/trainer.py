import copy
import logging
from pathlib import Path
from typing import List, Union

import torch
import torch.distributed
from torch.utils.data import DataLoader, DistributedSampler
import warnings
import metatensor.torch

from metatrain.utils.additive import remove_additive
from metatrain.utils.data import CombinedDataLoader, Dataset, _is_disk_dataset, CollateFn
from metatrain.utils.distributed.distributed_data_parallel import DistributedDataParallel
from metatrain.utils.distributed.slurm import DistributedEnvironment
from metatrain.utils.evaluate_model import evaluate_model
from metatrain.utils.external_naming import to_external_name
from metatrain.utils.io import check_file_extension
from metatrain.utils.logging import MetricLogger
#from metatrain.utils.loss import TensorMapDictLoss
from metatrain.utils.metrics import MAEAccumulator, RMSEAccumulator, get_selected_metric
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)
from metatrain.utils.per_atom import average_by_num_atoms
from metatrain.utils.scaler import remove_scale
#from metatrain.utils.transfer import (
#    systems_and_targets_to_device,
#    systems_and_targets_to_dtype,
#)
from .model import FlashMD
from .modules.augmentation import RotationalAugmenter
from metatensor.torch import Labels, TensorBlock, TensorMap
from ..utils.data import TargetInfo
from ..utils.data.target_info import get_energy_target_info

from metatomic.torch import load_atomistic_model, ModelOutput, ModelEvaluationOptions, System
from ase.data import atomic_masses

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, train_hypers):
        self.hypers = train_hypers
        self.optimizer_state_dict = None
        self.scheduler_state_dict = None
        self.epoch = None
        self.best_metric = None
        self.best_model_state_dict = None
        self.best_optimizer_state_dict = None

    def train(
        self,
        model: FlashMD,
        dtype: torch.dtype,
        devices: List[torch.device],
        train_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        val_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        checkpoint_dir: str,
    ):
        assert dtype in FlashMD.__supported_dtypes__

        is_distributed = self.hypers["distributed"]

        if is_distributed:
            distr_env = DistributedEnvironment(self.hypers["distributed_port"])
            torch.distributed.init_process_group(backend="nccl")
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
        else:
            rank = 0

        if is_distributed:
            if len(devices) > 1:
                raise ValueError(
                    "Requested distributed training with the `multi-gpu` device. "
                    " If you want to run distributed training with SOAP-BPNN, please "
                    "set `device` to cuda."
                )
            # the calculation of the device number works both when GPUs on different
            # processes are not visible to each other and when they are
            device_number = distr_env.local_rank % torch.cuda.device_count()
            device = torch.device("cuda", device_number)
        else:
            device = devices[
                0
            ]  # only one device, as we don't support multi-gpu for now

        if is_distributed:
            logger.info(f"Training on {world_size} devices with dtype {dtype}")
        else:
            logger.info(f"Training on device {device} with dtype {dtype}")

        # Calculate the neighbor lists in advance (in particular, this
        # needs to happen before the additive models are trained, as they
        # might need them):
        logger.info("Calculating neighbor lists for the datasets")
        requested_neighbor_lists = get_requested_neighbor_lists(model)
        for dataset in train_datasets + val_datasets:
            # If the dataset is a disk dataset, the NLs are already attached, we will
            # just check the first system
            if _is_disk_dataset(dataset):
                system = dataset[0]["system"]
                for options in requested_neighbor_lists:
                    if options not in system.known_neighbor_lists():
                        raise ValueError(
                            "The requested neighbor lists are not attached to the "
                            f"system. Neighbor list {options} is missing from the "
                            "first system in the disk dataset. Make sure you save "
                            "the neighbor lists in the systems when saving the dataset."
                        )
            else:
                for sample in dataset:
                    system = sample["system"]
                    # The following line attaches the neighbors lists to the system,
                    # and doesn't require to reassign the system to the dataset:
                    get_system_with_neighbor_lists(system, requested_neighbor_lists)

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
                train_datasets, model.additive_models, self.hypers["fixed_scaling_weights"], treat_as_additive=True
            )

        if is_distributed:
            model = DistributedDataParallel(model, device_ids=[device])

        logger.info("Setting up data loaders")

        if is_distributed:
            train_samplers = [
                DistributedSampler(
                    train_dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=True,
                    drop_last=True,
                )
                for train_dataset in train_datasets
            ]
            val_samplers = [
                DistributedSampler(
                    val_dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=False,
                    drop_last=False,
                )
                for val_dataset in val_datasets
            ]
        else:
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

        # Extract all the possible outputs and their gradients:
        train_targets = (model.module if is_distributed else model).scaler.dataset_info.targets
        outputs_list = []
        for target_name, target_info in train_targets.items():
            outputs_list.append(target_name)
            for gradient_name in target_info.gradients:
                outputs_list.append(f"{target_name}_{gradient_name}_gradients")
        # Create a loss weight dict:
        loss_weights_dict = {}
        for output_name in outputs_list:
            loss_weights_dict[output_name] = (
                self.hypers["loss"]["weights"][
                    to_external_name(output_name, train_targets)
                ]
                if to_external_name(output_name, train_targets)
                in self.hypers["loss"]["weights"]
                else 1.0
            )
        loss_weights_dict_external = {
            to_external_name(key, train_targets): value
            for key, value in loss_weights_dict.items()
        }
        loss_hypers = copy.deepcopy(self.hypers["loss"])
        loss_hypers["weights"] = loss_weights_dict
        logging.info(f"Training with loss weights: {loss_weights_dict_external}")

        # Create a loss function:
        loss_fn = TensorMapDictLoss(
            **loss_hypers,
        )

        # Create an optimizer:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.hypers["learning_rate"]
        )
        if self.optimizer_state_dict is not None:
            # try to load the optimizer state dict, but this is only possible
            # if there are no new targets in the model (new parameters)
            if not (model.module if is_distributed else model).has_new_targets:
                optimizer.load_state_dict(self.optimizer_state_dict)

        # Create a scheduler:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.hypers["scheduler_patience"],
            gamma=self.hypers["scheduler_factor"],
        )
        if self.scheduler_state_dict is not None:
            # same as the optimizer, try to load the scheduler state dict
            if not (model.module if is_distributed else model).has_new_targets:
                lr_scheduler.load_state_dict(self.scheduler_state_dict)

        # per-atom targets:
        per_structure_targets = self.hypers["per_structure_targets"]

        # Log the initial learning rate:
        old_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"Initial learning rate: {old_lr}")

        rotational_augmenter = RotationalAugmenter(train_targets)

        start_epoch = 0 if self.epoch is None else self.epoch + 1

        scaler_scales = (
            model.module if is_distributed else model
        ).scaler.get_scales_dict()

        potential_energy_model = load_atomistic_model(self.hypers["energy_model_path"]).to(device=device)
        evaluation_options = ModelEvaluationOptions(
            length_unit="Angstrom",
            outputs={"energy": ModelOutput(unit="eV")},
        )
        atomic_masses_torch = torch.tensor(atomic_masses, dtype=torch.float32, device="cuda")

        n_time_steps = int([tk for tk in (model.module if is_distributed else model).dataset_info.targets.keys() if "mtt::delta_" in tk and "_q" in tk][0].split("_")[1])

        # Train the model:
        if self.best_metric is None:
            self.best_metric = float("inf")
        logger.info("Starting training")
        epoch = start_epoch
        for epoch in range(start_epoch, start_epoch + self.hypers["num_epochs"]):
            if is_distributed:
                sampler.set_epoch(epoch)

            train_rmse_calculator = RMSEAccumulator(self.hypers["log_separate_blocks"])
            val_rmse_calculator = RMSEAccumulator(self.hypers["log_separate_blocks"])
            if self.hypers["log_mae"]:
                train_mae_calculator = MAEAccumulator(
                    self.hypers["log_separate_blocks"]
                )
                val_mae_calculator = MAEAccumulator(self.hypers["log_separate_blocks"])

            train_loss = 0.0
            for batch in train_dataloader:
                optimizer.zero_grad()

                systems, targets = batch
                systems, targets = rotational_augmenter.apply_random_augmentations(
                    systems, targets
                )
                systems, targets = systems_and_targets_to_device(
                    systems, targets, device
                )
                for additive_model in (
                    model.module if is_distributed else model
                ).additive_models:
                    targets = remove_additive(
                        systems, targets, additive_model, train_targets
                    )
                targets = remove_scale(
                    targets, (model.module if is_distributed else model).scaler
                )
                systems, targets = systems_and_targets_to_dtype(systems, targets, dtype)
                predictions = evaluate_model(
                    model,
                    systems,
                    {key: train_targets[key] for key in targets.keys() if "energy" not in key},
                    is_training=True,
                )
                rescaled_predictions = {k: metatensor.torch.multiply(p, scaler_scales[k]) for k, p in predictions.items()}
                
                ###
                predictions[f"mtt::energy_{n_time_steps}"] = fake_get_total_energy(systems, rescaled_predictions, potential_energy_model, evaluation_options, atomic_masses_torch)
                ###

                # average by the number of atoms
                predictions = average_by_num_atoms(
                    predictions, systems, per_structure_targets
                )
                targets = average_by_num_atoms(targets, systems, per_structure_targets)

                train_loss_batch = loss_fn(predictions, targets)
                train_loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                if is_distributed:
                    # sum the loss over all processes
                    torch.distributed.all_reduce(train_loss_batch)
                train_loss += train_loss_batch.item()
                train_rmse_calculator.update(predictions, targets)
                if self.hypers["log_mae"]:
                    train_mae_calculator.update(predictions, targets)

            finalized_train_info = train_rmse_calculator.finalize(
                not_per_atom=["positions_gradients"] + per_structure_targets,
                is_distributed=is_distributed,
                device=device,
            )
            if self.hypers["log_mae"]:
                finalized_train_info.update(
                    train_mae_calculator.finalize(
                        not_per_atom=["positions_gradients"] + per_structure_targets,
                        is_distributed=is_distributed,
                        device=device,
                    )
                )

            val_loss = 0.0
            with torch.no_grad():
                for batch in val_dataloader:
                    systems, targets = batch
                    systems = [system.to(device=device) for system in systems]
                    targets = {
                        key: value.to(device=device) for key, value in targets.items()
                    }
                    for additive_model in (
                        model.module if is_distributed else model
                    ).additive_models:
                        targets = remove_additive(
                            systems, targets, additive_model, train_targets
                        )
                    targets = remove_scale(
                        targets, (model.module if is_distributed else model).scaler
                    )
                    systems = [system.to(dtype=dtype) for system in systems]
                    targets = {key: value.to(dtype=dtype) for key, value in targets.items()}
                    predictions = evaluate_model(
                        model,
                        systems,
                        {key: train_targets[key] for key in targets.keys() if "energy" not in key},
                        is_training=False,
                    )

                    rescaled_predictions = {k: metatensor.torch.multiply(p, scaler_scales[k]) for k, p in predictions.items()}
                    
                    ###
                    predictions[f"mtt::energy_{n_time_steps}"] = get_total_energy(systems, rescaled_predictions, potential_energy_model, evaluation_options, atomic_masses_torch)
                    ###

                    # average by the number of atoms
                    predictions = average_by_num_atoms(
                        predictions, systems, per_structure_targets
                    )
                    targets = average_by_num_atoms(targets, systems, per_structure_targets)

                    val_loss_batch = loss_fn(predictions, targets)

                    if is_distributed:
                        # sum the loss over all processes
                        torch.distributed.all_reduce(val_loss_batch)
                    val_loss += val_loss_batch.item()
                    val_rmse_calculator.update(predictions, targets)
                    if self.hypers["log_mae"]:
                        val_mae_calculator.update(predictions, targets)

            finalized_val_info = val_rmse_calculator.finalize(
                not_per_atom=["positions_gradients"] + per_structure_targets,
                is_distributed=is_distributed,
                device=device,
            )
            if self.hypers["log_mae"]:
                finalized_val_info.update(
                    val_mae_calculator.finalize(
                        not_per_atom=["positions_gradients"] + per_structure_targets,
                        is_distributed=is_distributed,
                        device=device,
                    )
                )

            # Now we log the information:
            finalized_train_info = {"loss": train_loss, **finalized_train_info}
            finalized_val_info = {
                "loss": val_loss,
                **finalized_val_info,
            }

            if epoch == start_epoch:
                dataset_info_for_logging = copy.deepcopy(
                    model.module if is_distributed else model
                ).dataset_info

                ###
                dataset_info_for_logging.targets[f"mtt::energy_{n_time_steps}"] = get_energy_target_info({"unit": "eV", "per_atom": False})
                ###
                
                scaler_scales = (
                    model.module if is_distributed else model
                ).scaler.get_scales_dict()
                metric_logger = MetricLogger(
                    log_obj=logger,
                    dataset_info=dataset_info_for_logging,
                    initial_metrics=[finalized_train_info, finalized_val_info],
                    names=["training", "validation"],
                    scales={
                        key: (
                            scaler_scales[key.split(" ")[0]]
                            if ("MAE" in key or "RMSE" in key)
                            else 1.0
                        )
                        for key in finalized_train_info.keys()
                    },
                )
            if epoch % self.hypers["log_interval"] == 0:
                metric_logger.log(
                    metrics=[finalized_train_info, finalized_val_info],
                    epoch=epoch,
                    rank=rank,
                )

            lr_scheduler.step()
            new_lr = lr_scheduler.get_last_lr()[0]
            if new_lr != old_lr:
                if new_lr < 1e-7:
                    logger.info("Learning rate is too small, stopping training")
                    break
                else:
                    logger.info(f"Changing learning rate from {old_lr} to {new_lr}")
                    old_lr = new_lr
                    # load best model and optimizer state dict, re-initialize scheduler
                    (model.module if is_distributed else model).load_state_dict(
                        self.best_model_state_dict
                    )
                    optimizer.load_state_dict(self.best_optimizer_state_dict)
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = new_lr
                    lr_scheduler = torch.optim.lr_scheduler.StepLR(
                        optimizer,
                        step_size=self.hypers["scheduler_patience"],
                        gamma=self.hypers["scheduler_factor"],
                    )

            val_metric = get_selected_metric(
                finalized_val_info, self.hypers["best_model_metric"]
            )
            if val_metric < self.best_metric:
                self.best_metric = val_metric
                self.best_model_state_dict = copy.deepcopy(
                    (model.module if is_distributed else model).state_dict()
                )
                self.best_optimizer_state_dict = copy.deepcopy(optimizer.state_dict())

            if epoch % self.hypers["checkpoint_interval"] == 0:
                if is_distributed:
                    torch.distributed.barrier()
                self.optimizer_state_dict = optimizer.state_dict()
                self.scheduler_state_dict = lr_scheduler.state_dict()
                self.epoch = epoch
                if rank == 0:
                    self.save_checkpoint(
                        (model.module if is_distributed else model),
                        Path(checkpoint_dir) / f"model_{epoch}.ckpt",
                    )

        # prepare for the checkpoint that will be saved outside the function
        self.epoch = epoch
        self.optimizer_state_dict = optimizer.state_dict()
        self.scheduler_state_dict = lr_scheduler.state_dict()

    def save_checkpoint(self, model, path: Union[str, Path]):
        checkpoint = {
            "architecture_name": "experimental.nanopet",
            "model_data": {
                "model_hypers": model.hypers,
                "dataset_info": model.dataset_info,
            },
            "model_state_dict": model.state_dict(),
            "train_hypers": self.hypers,
            "epoch": self.epoch,
            "optimizer_state_dict": self.optimizer_state_dict,
            "scheduler_state_dict": self.scheduler_state_dict,
            "best_metric": self.best_metric,
            "best_model_state_dict": self.best_model_state_dict,
            "best_optimizer_state_dict": self.best_optimizer_state_dict,
        }
        torch.save(
            checkpoint,
            check_file_extension(path, ".ckpt"),
        )

    @classmethod
    def load_checkpoint(cls, path: Union[str, Path], train_hypers) -> "Trainer":
        # Load the checkpoint
        checkpoint = torch.load(path, weights_only=False)
        epoch = checkpoint["epoch"]
        optimizer_state_dict = checkpoint["optimizer_state_dict"]
        scheduler_state_dict = checkpoint["scheduler_state_dict"]
        best_metric = checkpoint["best_metric"]
        best_model_state_dict = checkpoint["best_model_state_dict"]
        best_optimizer_state_dict = checkpoint["best_optimizer_state_dict"]

        # Create the trainer
        trainer = cls(train_hypers)
        trainer.optimizer_state_dict = optimizer_state_dict
        trainer.scheduler_state_dict = scheduler_state_dict
        trainer.epoch = epoch
        trainer.best_metric = best_metric
        trainer.best_model_state_dict = best_model_state_dict
        trainer.best_optimizer_state_dict = best_optimizer_state_dict

        return trainer


def get_total_energy(systems, predictions, potential_energy_model, evaluation_options, atomic_masses_torch):
    for key in predictions.keys():
        if "mtt::delta_" in key and "_q" in key:
            delta_q_key = key
        if "mtt::p_" in key:
            p_prime_key = key
    
    delta_q = torch.split(predictions[delta_q_key].block().values.squeeze(-1), [len(system) for system in systems])
    p_prime = torch.split(predictions[p_prime_key].block().values.squeeze(-1), [len(system) for system in systems])
    # p_prime = [system.get_data("momenta").block().values.squeeze(-1) for system in systems]

    new_systems = []
    for dq, system in zip(delta_q, systems):
        new_system = System(
            positions=system.positions + dq / torch.sqrt(atomic_masses_torch[system.types].unsqueeze(-1)),
            types=system.types,
            cell=system.cell,
            pbc=system.pbc,
        )
        with warnings.catch_warnings():
            # this seems to be the only way to filter out the torch-scripted warnings
            # about neighbors (which are not relevant here), regex fails
            warnings.simplefilter("ignore")
            new_system = get_system_with_neighbor_lists(new_system, potential_energy_model.requested_neighbor_lists())
        new_systems.append(new_system)

    # new_systems = systems

    potential_energies = potential_energy_model(new_systems, evaluation_options, check_consistency=False)["energy"].block().values.squeeze(-1)
    kinetic_energies = [0.5 * torch.sum((p_prime[i] ** 2)) for i in range(len(systems))]
    kinetic_energies = torch.stack(kinetic_energies)
    total_energies = potential_energies + kinetic_energies
    return TensorMap(
        keys=Labels.single().to(device=total_energies.device),
        blocks=[
            TensorBlock(
                values=total_energies.unsqueeze(-1),
                samples=Labels(names=["system"], values=torch.arange(len(systems), device=total_energies.device).unsqueeze(-1)),
                components=[],
                properties=Labels.single().to(device=total_energies.device),
            )
        ],
    )

def fake_get_total_energy(systems, predictions, potential_energy_model, evaluation_options, atomic_masses_torch):

    tensor = torch.zeros((len(systems), 1), dtype=systems[0].positions.dtype, device=systems[0].device)

    return TensorMap(
        keys=Labels.single().to(device=tensor.device),
        blocks=[
            TensorBlock(
                values=tensor,
                samples=Labels(names=["system"], values=torch.arange(len(systems), device=tensor.device).unsqueeze(-1)),
                components=[],
                properties=Labels.single().to(device=tensor.device),
            )
        ],
    )
