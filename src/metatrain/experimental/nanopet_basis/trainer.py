import copy
import logging
from pathlib import Path
from typing import List, Union

import metatensor.torch as mts
import torch
import torch.distributed
from torch.utils.data import DataLoader, DistributedSampler

from ...utils.data import _is_disk_dataset
from ...utils.data.dataset import DatasetInfo
from ...utils.data.target_info import TargetInfo
from ...utils.data import CombinedDataLoader, Dataset, _is_disk_dataset, collate_fn
from ...utils.distributed.distributed_data_parallel import DistributedDataParallel
from ...utils.distributed.slurm import DistributedEnvironment
from ...utils.evaluate_model import evaluate_model
from ...utils.external_naming import to_external_name
from ...utils.io import check_file_extension
from ...utils.logging import MetricLogger
from ...utils.loss import TensorMapDictLoss
from ...utils.metrics import RMSEAccumulator, get_selected_metric
from ...utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)
from .utils import get_system_transformations  # , group_and_join_nonetypes  # TODO: collate needed?
from ...utils.transfer import (
    systems_and_targets_to_device,
    systems_and_targets_to_dtype,
)
from .model import NanoPETBasis
from ..nanopet.modules.augmentation import RotationalAugmenter


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
        model: NanoPETBasis,
        dtype: torch.dtype,
        devices: List[torch.device],
        train_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        val_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        checkpoint_dir: str,
        loss_fn: torch.nn.Module,
    ):
        assert dtype in NanoPETBasis.__supported_dtypes__

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

        train_targets = (model.module if is_distributed else model).dataset_info.targets
        
        # Create a loss function:
        if loss_fn is None:
            loss_fn = TensorMapDictLoss(**self.hypers["loss"])
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
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=self.hypers["scheduler_factor"],
            patience=self.hypers["scheduler_patience"],
        )
        if self.scheduler_state_dict is not None:
            # same as the optimizer, try to load the scheduler state dict
            if not (model.module if is_distributed else model).has_new_targets:
                lr_scheduler.load_state_dict(self.scheduler_state_dict)

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
        for epoch in range(start_epoch, start_epoch + self.hypers["num_epochs"]):
            if is_distributed:
                sampler.set_epoch(epoch)

            train_rmse_calculator = RMSEAccumulator(self.hypers["log_separate_blocks"])
            val_rmse_calculator = RMSEAccumulator(self.hypers["log_separate_blocks"])

            model.train()
            train_loss = 0.0
            for batch in train_dataloader:
                optimizer.zero_grad()

                systems_, targets = batch
                systems_, targets = systems_and_targets_to_device(
                    systems_, targets, device
                )
                systems_, targets = systems_and_targets_to_dtype(
                    systems_, targets, dtype
                )

                # TODO: remove sorting!
                targets["node"] = mts.sort(targets["node"])
                targets["edge"] = mts.sort(targets["edge"])

                # TODO: use `evaluate_model` instead?
                # Define a random transformation for each training system. This needs to
                # be done outside of the augmenter as the same transformations need to
                # be applied to both nodes and edges.
                rotations, inversions = get_system_transformations(systems_)

                # Apply rotational augmentation - node
                if model.in_keys_node is not None:
                    # TODO: pass both node and edges here?
                    systems, targets_node = (
                        rotational_augmenter.apply_augmentations(
                            systems_,
                            {"node": targets["node"]},
                            rotations,
                            inversions,
                        )
                    )
                    targets_node = targets_node["node"]
                    assert mts.equal_metadata(targets["node"], targets_node)
                    assert not mts.allclose(targets["node"], targets_node)
                    targets["node"] = targets_node

                # Apply rotational augmentation - edge
                if model.in_keys_edge is not None:
                    _, targets_edge = rotational_augmenter.apply_augmentations(
                        systems_,
                        {"edge": targets["edge"]},
                        rotations,
                        inversions,
                    )
                    targets_edge = targets_edge["edge"]
                    assert mts.equal_metadata(targets["edge"], targets_edge)
                    assert not mts.allclose(targets["edge"], targets_edge)
                    targets["edge"] = targets_edge

                predictions = model(systems, {}, selected_atoms=None)

                # TODO: remove sorting!
                predictions["node"] = mts.sort(predictions["node"])
                predictions["edge"] = mts.sort(predictions["edge"])

                train_loss_batch = loss_fn(
                    predictions, targets, self.hypers["loss"]["weights"],
                )
                train_loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                if is_distributed:
                    # sum the loss over all processes
                    torch.distributed.all_reduce(train_loss_batch)
                train_loss += train_loss_batch.item()
                train_rmse_calculator.update(predictions, targets)

            finalized_train_info = train_rmse_calculator.finalize(
                ["node", "edge"],
                device=device,
            )

            val_loss = 0.0
            for batch in val_dataloader:
                systems, targets = batch
                systems_, targets = systems_and_targets_to_device(
                    systems_, targets, device
                )
                
                # TODO: remove sorting!
                targets["node"] = mts.sort(targets["node"])
                targets["edge"] = mts.sort(targets["edge"])

                model.eval()
                node_predictions, edge_predictions = model(systems, {}, None)
                
                # TODO: remove sorting!
                predictions_node = mts.sort(predictions_node)
                predictions_edge = mts.sort(predictions_edge)
                predictions = {"node": node_predictions, "edge": edge_predictions}
                targets = {"node": targets["node"], "edge": targets["edge"]}

                val_loss_batch = loss_fn(
                    predictions, targets, self.hypers["loss"]["weights"]
                )
                val_loss += val_loss_batch.item()
                val_rmse_calculator.update(predictions, targets)

            finalized_val_info = val_rmse_calculator.finalize(
                ["node", "edge"], device=device
            )

            finalized_train_info = {"loss": train_loss, **finalized_train_info}
            finalized_val_info = {
                "loss": val_loss,
                **finalized_val_info,
            }

            if epoch == start_epoch:
                metric_logger = MetricLogger(
                    log_obj=logger,
                    dataset_info=(
                        model.module if is_distributed else model
                    ).dataset_info,
                    initial_metrics=[finalized_train_info, finalized_val_info],
                    names=["training", "validation"],
                )
            if epoch % self.hypers["log_interval"] == 0:
                metric_logger.log(
                    metrics=[finalized_train_info, finalized_val_info],
                    epoch=epoch,
                    rank=rank,
                )

            lr_scheduler.step(val_loss)
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
                    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        factor=self.hypers["scheduler_factor"],
                        patience=self.hypers["scheduler_patience"],
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
            "architecture_name": "experimental.nanopet_basis",
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
