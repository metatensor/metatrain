import copy
import logging
import warnings
from pathlib import Path
from typing import List, Union

import torch
import torch.distributed
from torch.utils.data import DataLoader, DistributedSampler

from ...utils.data import CombinedDataLoader, Dataset, collate_fn
from ...utils.distributed.distributed_data_parallel import DistributedDataParallel
from ...utils.distributed.slurm import DistributedEnvironment
from ...utils.evaluate_model import evaluate_model
from ...utils.external_naming import to_external_name
from ...utils.io import check_file_extension
from ...utils.logging import MetricLogger
from ...utils.loss import TensorMapDictLoss
from ...utils.metrics import MAEAccumulator, RMSEAccumulator
from ...utils.neighbor_lists import get_system_with_neighbor_lists
from ...utils.per_atom import average_by_num_atoms
from .model import PhACE
from ...utils.scaler import remove_scale
from ...utils.transfer import (
    systems_and_targets_to_device,
    systems_and_targets_to_dtype,
)
from ...utils.additive import remove_additive


logger = logging.getLogger(__name__)


# Filter out the second derivative and device warnings from rascaline-torch
warnings.filterwarnings("ignore", category=UserWarning, message="second derivative")
warnings.filterwarnings(
    "ignore", category=UserWarning, message="Systems data is on device"
)


class Trainer:
    def __init__(self, train_hypers):
        self.hypers = train_hypers
        self.optimizer_state_dict = None
        self.scheduler_state_dict = None
        self.epoch = None
        self.best_loss = None
        self.best_model_state_dict = None
        self.best_optimizer_state_dict = None

    def train(
        self,
        model: PhACE,
        dtype: torch.dtype,
        devices: List[torch.device],
        train_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        val_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        checkpoint_dir: str,
    ):
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

        # Move the model to the device and dtype:
        model.to(device=device, dtype=dtype)
        # The additive models of the SOAP-BPNN are always in float64 (to avoid
        # numerical errors in the composition weights, which can be very large).
        for additive_model in model.additive_models:
            additive_model.to(dtype=torch.float64)

        logger.info("Calculating composition weights")
        model.additive_models[0].train_model(  # this is the composition model
            train_datasets, self.hypers["fixed_composition_weights"]
        )

        # Calculate NLs:
        logger.info("Calculating neighbors lists for the datasets")
        requested_neighbor_lists = model.requested_neighbor_lists()
        for dataset in train_datasets + val_datasets:
            for i in range(len(dataset)):
                system = dataset[i]["system"]
                # The following line attached the neighbors lists to the system,
                # and doesn't require to reassign the system to the dataset:
                _ = get_system_with_neighbor_lists(system, requested_neighbor_lists)

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
        train_targets = (model.module if is_distributed else model).dataset_info.targets
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

        if self.hypers["scale_targets"]:
            logger.info("Calculating scaling weights")
            model.scaler.train_model(train_datasets, model.additive_models)

        torch.jit.set_fusion_strategy([("DYNAMIC", 0)])
        scripted_model = torch.jit.script(model)
        if is_distributed:
            scripted_model = DistributedDataParallel(
                scripted_model, device_ids=[device], find_unused_parameters=False
            )

        # Create an optimizer:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.hypers["learning_rate"]
        )
        if self.optimizer_state_dict is not None:
            # try to load the optimizer state dict, but this is only possible
            # if there are no new targets in the model (new parameters)
            if not model.has_new_targets:
                optimizer.load_state_dict(self.optimizer_state_dict)

        # Create an optimizer and a scheduler:
        optimizer = torch.optim.AdamW(
            scripted_model.parameters(),
            lr=self.hypers["learning_rate"],
            amsgrad=True,
            weight_decay=5e-5,
        )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=self.hypers["scheduler_factor"],
            patience=self.hypers["scheduler_patience"],
            threshold=0.001,
        )
        if self.scheduler_state_dict is not None:
            # same as the optimizer, try to load the scheduler state dict
            if not model.has_new_targets:
                lr_scheduler.load_state_dict(self.scheduler_state_dict)

        # per-atom targets:
        per_structure_targets = self.hypers["per_structure_targets"]

        # Log the initial learning rate:
        old_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"Initial learning rate: {old_lr}")

        # Train the model:
        if self.best_loss is None:
            self.best_loss = float("inf")
        logger.info("Starting training")

        for epoch in range(self.hypers["num_epochs"]):
            train_rmse_calculator = RMSEAccumulator()
            train_mae_calculator = MAEAccumulator()
            val_rmse_calculator = RMSEAccumulator()
            val_mae_calculator = MAEAccumulator()

            train_loss = 0.0
            for batch in train_dataloader:
                optimizer.zero_grad()

                systems, targets = batch
                systems, targets = batch
                systems, targets = systems_and_targets_to_device(
                    systems, targets, device
                )
                for additive_model in (
                    model.module if is_distributed else model
                ).additive_models:
                    targets = remove_additive(
                        systems, targets, additive_model, train_targets
                    )
                targets = remove_scale(targets, (
                    model.module if is_distributed else model
                ).scaler)
                systems, targets = systems_and_targets_to_dtype(systems, targets, dtype)
                predictions = evaluate_model(
                    scripted_model,
                    systems,
                    {key: train_targets[key] for key in targets.keys()},
                    is_training=True,
                )

                # average by the number of atoms
                predictions = average_by_num_atoms(
                    predictions, systems, per_structure_targets
                )
                targets = average_by_num_atoms(targets, systems, per_structure_targets)

                train_loss_batch = loss_fn(predictions, targets)
                train_loss += train_loss_batch.item()
                train_loss_batch.backward()
                optimizer.step()
                train_rmse_calculator.update(predictions, targets)
                train_mae_calculator.update(predictions, targets)

            finalized_train_info = {
                **train_rmse_calculator.finalize(
                    not_per_atom=["positions_gradients"] + per_structure_targets
                ),
                **train_mae_calculator.finalize(
                    not_per_atom=["positions_gradients"] + per_structure_targets
                ),
            }

            val_loss = 0.0
            for batch in val_dataloader:
                systems, targets = batch
                systems, targets = systems_and_targets_to_device(
                    systems, targets, device
                )
                for additive_model in (
                    model.module if is_distributed else model
                ).additive_models:
                    targets = remove_additive(
                        systems, targets, additive_model, train_targets
                    )
                targets = remove_scale(targets, (
                    model.module if is_distributed else model
                ).scaler)
                systems, targets = systems_and_targets_to_dtype(systems, targets, dtype)
                predictions = evaluate_model(
                    scripted_model,
                    systems,
                    {key: train_targets[key] for key in targets.keys()},
                    is_training=False,
                )

                # average by the number of atoms
                predictions = average_by_num_atoms(
                    predictions, systems, per_structure_targets
                )
                targets = average_by_num_atoms(targets, systems, per_structure_targets)

                val_loss_batch = loss_fn(predictions, targets)
                val_loss += val_loss_batch.item()
                val_rmse_calculator.update(predictions, targets)
                val_mae_calculator.update(predictions, targets)

            finalized_val_info = {
                **val_rmse_calculator.finalize(
                    not_per_atom=["positions_gradients"] + per_structure_targets
                ),
                **val_mae_calculator.finalize(
                    not_per_atom=["positions_gradients"] + per_structure_targets
                ),
            }

            # Now we log the information:
            finalized_train_info = {"loss": train_loss, **finalized_train_info}
            finalized_val_info = {
                "loss": val_loss,
                **finalized_val_info,
            }

            if epoch == 0:
                scaler_scales = (
                    model.module if is_distributed else model
                ).scaler.get_scales_dict()
                metric_logger = MetricLogger(
                    log_obj=logger,
                    dataset_info=(
                        model.module if is_distributed else model
                    ).dataset_info,
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
                )

            # if epoch % self.hypers["checkpoint_interval"] == 0:
            #     model.save_checkpoint(Path(checkpoint_dir) / f"model_{epoch}.ckpt")
            # TODO: how do I make this work given that it's scripted?

            # lr_before = optimizer.param_groups[0]["lr"]
            # scheduler.step(val_loss)
            # lr_after = optimizer.param_groups[0]["lr"]
            # if lr_before != lr_after:
            #     logger.info(f"Learning rate changed from {lr_before} to {lr_after}")
            #     scripted_model.load_state_dict(best_state_dict)
            #     optimizer.load_state_dict(best_optimizer_state_dict)
            #     for param_group in optimizer.param_groups:
            #         param_group["lr"] = lr_after
            # if lr_after < 1e-6:
            #     logger.info("Training has converged, stopping")
            #     break

            # val_metric = (
            #     finalized_val_info["energy MAE (per atom)"]
            #     * finalized_val_info["energy_positions_gradients MAE"]
            #     if "energy_positions_gradients MAE" in finalized_train_info
            #     else (
            #         finalized_val_info["energy MAE"]
            #         if "energy MAE" in finalized_val_info
            #         else finalized_val_info["energy MAE (per atom)"]
            #     )
            # )
            lr_scheduler.step(val_loss)
            new_lr = lr_scheduler.get_last_lr()[0]
            if new_lr != old_lr:
                if new_lr < 1e-7:
                    logger.info("Learning rate is too small, stopping training")
                    break
                else:
                    logger.info(f"Changing learning rate from {old_lr} to {new_lr}")
                    old_lr = new_lr
                    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        mode="min",
                        factor=self.hypers["scheduler_factor"],
                        patience=self.hypers["scheduler_patience"],
                        threshold=0.001,
                    )

        # prepare for the checkpoint that will be saved outside the function
        self.optimizer_state_dict = optimizer.state_dict()
        self.scheduler_state_dict = lr_scheduler.state_dict()

        # if is_distributed:
        #     model.load_state_dict(
        #         {
        #             name.replace("module.", ""): tensor
        #             for name, tensor in best_state_dict.items()
        #         }
        #     )
        # else:
        #     model.load_state_dict(best_state_dict)

    def save_checkpoint(self, model, path: Union[str, Path]):
        # ???????????????
        checkpoint = {
            "model_hypers": {
                "model_hypers": model.hypers,
                "dataset_info": model.dataset_info,
            },
            "model_state_dict": model.state_dict(),
            "train_hypers": self.hypers,
            "epoch": self.epoch,
            "optimizer_state_dict": self.optimizer_state_dict,
            "scheduler_state_dict": self.scheduler_state_dict,
            "best_loss": self.best_loss,
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
        checkpoint = torch.load(path)
        epoch = checkpoint["epoch"]
        optimizer_state_dict = checkpoint["optimizer_state_dict"]
        scheduler_state_dict = checkpoint["scheduler_state_dict"]
        best_loss = checkpoint["best_loss"]
        best_model_state_dict = checkpoint["best_model_state_dict"]
        best_optimizer_state_dict = checkpoint["best_optimizer_state_dict"]

        # Create the trainer
        trainer = cls(train_hypers)
        trainer.optimizer_state_dict = optimizer_state_dict
        trainer.scheduler_state_dict = scheduler_state_dict
        trainer.epoch = epoch
        trainer.best_loss = best_loss
        trainer.best_model_state_dict = best_model_state_dict
        trainer.best_optimizer_state_dict = best_optimizer_state_dict

        return trainer
