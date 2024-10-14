import logging
import warnings
from pathlib import Path
from typing import List, Union

import torch
import torch.distributed
from torch.utils.data import DataLoader, DistributedSampler

from ...utils.additive import remove_additive
from ...utils.data import CombinedDataLoader, Dataset, TargetInfoDict, collate_fn
from ...utils.data.extract_targets import get_targets_dict
from ...utils.distributed.distributed_data_parallel import DistributedDataParallel
from ...utils.distributed.slurm import DistributedEnvironment
from ...utils.evaluate_model import evaluate_model
from ...utils.external_naming import to_external_name
from ...utils.io import check_file_extension
from ...utils.logging import MetricLogger
from ...utils.loss import TensorMapDictLoss
from ...utils.metrics import RMSEAccumulator
from ...utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)
from ...utils.per_atom import average_by_num_atoms
from .model import SoapBpnn


logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, train_hypers):
        self.hypers = train_hypers
        self.optimizer_state_dict = None
        self.scheduler_state_dict = None
        self.epoch = None

    def train(
        self,
        model: SoapBpnn,
        dtype: torch.dtype,
        devices: List[torch.device],
        train_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        val_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        checkpoint_dir: str,
    ):
        # Filter out the second derivative and device warnings from rascaline
        warnings.filterwarnings(action="ignore", message="Systems data is on device")
        warnings.filterwarnings(
            action="ignore",
            message="second derivatives with respect to positions are not implemented",
        )
        warnings.filterwarnings(
            action="ignore",
            message="second derivatives with respect to cell matrix",
        )

        assert dtype in SoapBpnn.__supported_dtypes__

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
            for i in range(len(dataset)):
                system = dataset[i]["system"]
                # The following line attaches the neighbors lists to the system,
                # and doesn't require to reassign the system to the dataset:
                _ = get_system_with_neighbor_lists(system, requested_neighbor_lists)

        # Move the model to the device and dtype:
        model.to(device=device, dtype=dtype)
        # The additive models of the SOAP-BPNN are always on CPU (to avoid OOM
        # errors during the linear algebra training) and in float64 (to avoid
        # numerical errors in the composition weights, which can be very large).
        for additive_model in model.additive_models:
            additive_model.to(device=torch.device("cpu"), dtype=torch.float64)

        logger.info("Calculating composition weights")
        model.additive_models[0].train_model(  # this is the composition model
            train_datasets, self.hypers["fixed_composition_weights"]
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
        train_targets = get_targets_dict(
            train_datasets, (model.module if is_distributed else model).dataset_info
        )
        outputs_list = []
        for target_name, target_info in train_targets.items():
            outputs_list.append(target_name)
            for gradient_name in target_info.gradients:
                outputs_list.append(f"{target_name}_{gradient_name}_gradients")
        # Create a loss weight dict:
        loss_weights_dict = {}
        for output_name in outputs_list:
            loss_weights_dict[output_name] = (
                self.hypers["loss_weights"][
                    to_external_name(output_name, train_targets)
                ]
                if to_external_name(output_name, train_targets)
                in self.hypers["loss_weights"]
                else 1.0
            )
        loss_weights_dict_external = {
            to_external_name(key, train_targets): value
            for key, value in loss_weights_dict.items()
        }
        logging.info(f"Training with loss weights: {loss_weights_dict_external}")

        # Create a loss function:
        loss_fn = TensorMapDictLoss(loss_weights_dict)

        # Create an optimizer:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.hypers["learning_rate"]
        )
        if self.optimizer_state_dict is not None:
            optimizer.load_state_dict(self.optimizer_state_dict)

        # Create a scheduler:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.hypers["scheduler_factor"],
            patience=self.hypers["scheduler_patience"],
        )
        if self.scheduler_state_dict is not None:
            lr_scheduler.load_state_dict(self.scheduler_state_dict)

        # counters for early stopping:
        best_val_loss = float("inf")
        epochs_without_improvement = 0

        # per-atom targets:
        per_structure_targets = self.hypers["per_structure_targets"]

        start_epoch = 0 if self.epoch is None else self.epoch + 1

        # Train the model:
        logger.info("Starting training")
        for epoch in range(start_epoch, start_epoch + self.hypers["num_epochs"]):
            if is_distributed:
                sampler.set_epoch(epoch)

            train_rmse_calculator = RMSEAccumulator()
            val_rmse_calculator = RMSEAccumulator()

            train_loss = 0.0
            for batch in train_dataloader:
                optimizer.zero_grad()

                systems, targets = batch
                for additive_model in model.additive_models:
                    targets = remove_additive(
                        systems, targets, additive_model, train_targets
                    )
                systems = [system.to(dtype=dtype, device=device) for system in systems]
                targets = {
                    key: value.to(dtype=dtype, device=device)
                    for key, value in targets.items()
                }
                predictions = evaluate_model(
                    model,
                    systems,
                    TargetInfoDict(
                        **{key: train_targets[key] for key in targets.keys()}
                    ),
                    is_training=True,
                )

                # average by the number of atoms
                predictions = average_by_num_atoms(
                    predictions, systems, per_structure_targets
                )
                targets = average_by_num_atoms(targets, systems, per_structure_targets)

                train_loss_batch = loss_fn(predictions, targets)

                train_loss_batch.backward()
                optimizer.step()

                if is_distributed:
                    # sum the loss over all processes
                    torch.distributed.all_reduce(train_loss_batch)
                train_loss += train_loss_batch.item()
                train_rmse_calculator.update(predictions, targets)
            finalized_train_info = train_rmse_calculator.finalize(
                not_per_atom=["positions_gradients"] + per_structure_targets,
                is_distributed=is_distributed,
                device=device,
            )

            val_loss = 0.0
            for batch in val_dataloader:
                systems, targets = batch
                for additive_model in model.additive_models:
                    targets = remove_additive(
                        systems, targets, additive_model, train_targets
                    )
                systems = [system.to(dtype=dtype, device=device) for system in systems]
                targets = {
                    key: value.to(dtype=dtype, device=device)
                    for key, value in targets.items()
                }
                predictions = evaluate_model(
                    model,
                    systems,
                    TargetInfoDict(
                        **{key: train_targets[key] for key in targets.keys()}
                    ),
                    is_training=False,
                )

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
            finalized_val_info = val_rmse_calculator.finalize(
                not_per_atom=["positions_gradients"] + per_structure_targets,
                is_distributed=is_distributed,
                device=device,
            )

            lr_scheduler.step(val_loss)

            # Now we log the information:
            finalized_train_info = {"loss": train_loss, **finalized_train_info}
            finalized_val_info = {
                "loss": val_loss,
                **finalized_val_info,
            }

            if epoch == start_epoch:
                metric_logger = MetricLogger(
                    log_obj=logger,
                    dataset_info=model.dataset_info,
                    initial_metrics=[finalized_train_info, finalized_val_info],
                    names=["training", "validation"],
                )
            if epoch % self.hypers["log_interval"] == 0:
                metric_logger.log(
                    metrics=[finalized_train_info, finalized_val_info],
                    epoch=epoch,
                    rank=rank,
                )

            if epoch % self.hypers["checkpoint_interval"] == 0:
                if is_distributed:
                    torch.distributed.barrier()
                self.optimizer_state_dict = optimizer.state_dict()
                self.scheduler_state_dict = lr_scheduler.state_dict()
                self.epoch = epoch
                self.save_checkpoint(
                    (model.module if is_distributed else model),
                    Path(checkpoint_dir) / f"model_{epoch}.ckpt",
                )

            # early stopping criterion:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.hypers["early_stopping_patience"]:
                    logger.info(
                        "Early stopping criterion reached after "
                        f"{self.hypers['early_stopping_patience']} epochs "
                        "without improvement."
                    )
                    break

    def save_checkpoint(self, model, path: Union[str, Path]):
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
        }
        torch.save(
            checkpoint,
            check_file_extension(path, ".ckpt"),
        )

    @classmethod
    def load_checkpoint(cls, path: Union[str, Path], train_hypers) -> "Trainer":

        # Load the checkpoint
        checkpoint = torch.load(path, weights_only=False)
        model_hypers = checkpoint["model_hypers"]
        model_state_dict = checkpoint["model_state_dict"]
        epoch = checkpoint["epoch"]
        optimizer_state_dict = checkpoint["optimizer_state_dict"]
        scheduler_state_dict = checkpoint["scheduler_state_dict"]

        # Create the trainer
        trainer = cls(train_hypers)
        trainer.optimizer_state_dict = optimizer_state_dict
        trainer.scheduler_state_dict = scheduler_state_dict
        trainer.epoch = epoch

        # Create the model
        model = SoapBpnn(**model_hypers)
        model.load_state_dict(model_state_dict)

        return trainer
