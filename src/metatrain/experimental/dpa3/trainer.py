import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Union, cast

import torch
import torch.distributed
from torch.utils.data import DistributedSampler

from metatrain.composition import train_or_load_composition_model
from metatrain.utils.abc import TrainerInterface
from metatrain.utils.additive import remove_additive
from metatrain.utils.data import (
    CollateFn,
    CombinedDataLoader,
    Dataset,
    _is_disk_dataset,
    build_train_dataloaders,
    build_val_dataloaders,
    unpack_batch,
)
from metatrain.utils.distributed.distributed_data_parallel import (
    DistributedDataParallel,
)
from metatrain.utils.distributed.slurm import (
    initialize_slurm_nccl_process_group,
    resolve_distributed,
)
from metatrain.utils.evaluate_model import evaluate_model
from metatrain.utils.io import check_file_extension
from metatrain.utils.logging import ROOT_LOGGER, MetricLogger
from metatrain.utils.loss import LossAggregator, LossSpecification
from metatrain.utils.metrics import (
    MAEAccumulator,
    RMSEAccumulator,
    get_selected_metric,
)
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)
from metatrain.utils.per_atom import average_by_num_atoms
from metatrain.utils.scaler import remove_scale
from metatrain.utils.transfer import (
    batch_to,
)

from . import checkpoints
from .documentation import TrainerHypers
from .model import DPA3


# Learning rate below which training is stopped early.
_MIN_LEARNING_RATE = 1e-7


def _get_raw_model(model: Union[DPA3, DistributedDataParallel], is_distributed: bool):
    """Unwrap a possibly-DDP-wrapped model to get the underlying DPA3."""
    return model.module if is_distributed else model


class Trainer(TrainerInterface[TrainerHypers]):
    __checkpoint_version__ = 2

    def __init__(self, hypers: TrainerHypers):
        super().__init__(hypers)

        self.optimizer_state_dict = None
        self.scheduler_state_dict = None
        self.epoch: int | None = None
        self.best_epoch: int | None = None
        self.best_metric: float | None = None
        self.best_model_state_dict = None
        self.best_optimizer_state_dict = None

    def train(
        self,
        model: DPA3,
        dtype: torch.dtype,
        devices: List[torch.device],
        train_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        val_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        checkpoint_dir: str,
    ):
        assert dtype in DPA3.__supported_dtypes__

        is_distributed = resolve_distributed(self.hypers["distributed"])

        if is_distributed:
            device, world_size, rank = initialize_slurm_nccl_process_group(
                self.hypers["distributed_port"]
            )
        else:
            rank = 0
            world_size = 1

        if is_distributed:
            if len(devices) > 1:
                raise ValueError(
                    "Requested distributed training with the `multi-gpu` device. "
                    " If you want to run distributed training with DPA3, please "
                    "set `device` to cuda."
                )
        else:
            device = devices[
                0
            ]  # only one device, as we don't support multi-gpu for now

        if is_distributed:
            logging.info(f"Training on {world_size} devices with dtype {dtype}")
        else:
            logging.info(f"Training on device {device} with dtype {dtype}")

        # Calculate the neighbor lists in advance (in particular, this
        # needs to happen before the additive models are trained, as they
        # might need them):
        logging.info("Calculating neighbor lists for the datasets")
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

        # Validate that base_precision matches the model's construction-time
        # precision.  deepmd-kit's internal self.prec is set at construction
        # and is NOT updated by .to(dtype=...), so a mismatch would cause
        # silent precision loss.
        model_dtype = next(model.model.parameters()).dtype
        if dtype != model_dtype:
            from .model import _PRECISION_INT_TO_DTYPE

            expected_prec = {v: k for k, v in _PRECISION_INT_TO_DTYPE.items()}.get(
                model_dtype, model_dtype
            )
            raise ValueError(
                f"base_precision ({dtype}) does not match the DPA3 model's "
                f"construction-time precision ({model_dtype}). Set "
                f"base_precision: {expected_prec} to match "
                f"descriptor.precision."
            )

        # Move the model to the device (dtype already matches construction):
        model.to(device=device)
        # The additive models are always kept in float64 to avoid numerical
        # errors in the composition weights, which can be very large.
        for additive_model in model.additive_models:
            additive_model.to(dtype=torch.float64)

        train_or_load_composition_model(
            composition_model=model.additive_models[0],
            atomic_baseline={
                **model.get_fixed_composition_weights(),
                **self.hypers["fixed_composition_weights"],
            },
            train_datasets=train_datasets,
            other_additive_models=list(model.additive_models[1:]),
            batch_size=self.hypers["batch_size"],
            is_distributed=is_distributed,
            checkpoint_dir=checkpoint_dir,
        )

        if self.hypers["scale_targets"]:
            logging.info("Calculating scaling weights")
            model.scaler.train_model(
                train_datasets,
                model.additive_models,
                self.hypers["batch_size"],
                is_distributed,
                model.get_fixed_scaling_weights(),
                per_structure_targets=self.hypers["per_structure_targets"],
            )

        if is_distributed:
            model = DistributedDataParallel(model, device_ids=[device])

        raw_model = _get_raw_model(model, is_distributed)

        logging.info("Setting up data loaders")

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

        # Create a collate function:
        targets_keys = list(raw_model.dataset_info.targets.keys())
        collate_fn = CollateFn(target_keys=targets_keys)

        max_atoms = self.hypers["max_atoms_per_batch"]

        # Create dataloader for the training datasets:
        train_dataloaders, epoch_samplers = build_train_dataloaders(
            train_datasets=train_datasets,
            train_distributed_samplers=train_samplers,
            collate_fn_train=collate_fn,
            batch_size=self.hypers["batch_size"],
            max_atoms_per_batch=max_atoms,
            min_atoms_per_batch=self.hypers["min_atoms_per_batch"],
            num_workers=0,
        )
        train_dataloader = CombinedDataLoader(train_dataloaders, shuffle=True)

        # Create dataloader for the validation datasets:
        val_dataloaders = build_val_dataloaders(
            val_datasets=val_datasets,
            val_distributed_samplers=val_samplers,
            collate_fn_val=collate_fn,
            batch_size=self.hypers["batch_size"],
            max_atoms_per_batch=max_atoms,
            num_workers=0,
        )
        val_dataloader = CombinedDataLoader(val_dataloaders, shuffle=False)

        # Extract all the possible outputs and their gradients:
        train_targets = raw_model.dataset_info.targets
        outputs_list = []
        for target_name, target_info in train_targets.items():
            outputs_list.append(target_name)
            for gradient_name in target_info.gradients:
                outputs_list.append(f"{target_name}_{gradient_name}_gradients")

        # Create a loss function:
        loss_hypers = cast(Dict[str, LossSpecification], self.hypers["loss"])  # mypy
        loss_fn = LossAggregator(
            targets=train_targets,
            config=loss_hypers,
        )
        logging.info("Using the following loss functions:")
        for name, info in loss_fn.metadata.items():
            logging.info(f"{name}:")
            main = {k: v for k, v in info.items() if k != "gradients"}
            logging.info(main)
            if "gradients" not in info or len(info["gradients"]) == 0:
                continue
            logging.info("With gradients:")
            for grad, ginfo in info["gradients"].items():
                logging.info(f"\t{name}::{grad}: {ginfo}")

        # Create an optimizer:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.hypers["learning_rate"]
        )
        if self.optimizer_state_dict is not None:
            # try to load the optimizer state dict, but this is only possible
            # if there are no new targets in the model (new parameters)
            if not raw_model.has_new_targets:
                optimizer.load_state_dict(self.optimizer_state_dict)

        # Create a scheduler:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=self.hypers["scheduler_factor"],
            patience=self.hypers["scheduler_patience"],
            threshold=0.001,
            min_lr=1e-5,
        )
        if self.scheduler_state_dict is not None:
            # same as the optimizer, try to load the scheduler state dict
            if not raw_model.has_new_targets:
                lr_scheduler.load_state_dict(self.scheduler_state_dict)

        # per-atom targets:
        per_structure_targets = self.hypers["per_structure_targets"]

        # Log the initial learning rate:
        old_lr = optimizer.param_groups[0]["lr"]
        logging.info(f"Initial learning rate: {old_lr}")

        start_epoch = 0 if self.epoch is None else self.epoch + 1

        # Train the model:
        if self.best_metric is None:
            self.best_metric = float("inf")
        logging.info("Starting training")
        epoch = start_epoch
        for epoch in range(start_epoch, start_epoch + self.hypers["num_epochs"]):
            for sampler in epoch_samplers:
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

                systems, targets, extra_data = unpack_batch(batch)
                systems, targets, extra_data = batch_to(
                    systems, targets, extra_data, device=device
                )
                for additive_model in raw_model.additive_models:
                    targets = remove_additive(
                        systems, targets, additive_model, train_targets
                    )
                targets = remove_scale(systems, targets, raw_model.scaler)
                systems, targets, extra_data = batch_to(
                    systems, targets, extra_data, dtype=dtype
                )

                predictions = evaluate_model(
                    model,
                    systems,
                    {key: train_targets[key] for key in targets.keys()},
                    is_training=True,
                )

                # average by the number of atoms
                predictions = average_by_num_atoms(
                    predictions, systems, per_structure_targets
                )
                targets = average_by_num_atoms(targets, systems, per_structure_targets)

                # Apply per-property scales to the predictions before loss computation.
                # The targets from the dataloader have only been scaled per-target, and
                # not per-property. This transformation only applies to targets with
                # per-property scales (i.e. multiple blocks or multiple properties), and
                # leaves the others unchanged.
                predictions = (model.module if is_distributed else model).scaler(
                    systems,
                    predictions,
                    remove=False,
                    use_per_target_scales=False,  # never before loss
                    use_per_property_scales=True,
                )

                train_loss_batch = loss_fn(predictions, targets, extra_data)

                train_loss_batch.backward()
                optimizer.step()

                if is_distributed:
                    # sum the loss over all processes
                    torch.distributed.all_reduce(train_loss_batch)
                train_loss += train_loss_batch.item()

                # Reapply scales and accumulate quantities for computing train metrics,
                # but only if this is an epoch to log
                if epoch == start_epoch or epoch % self.hypers["log_interval"] == 0:
                    scaled_predictions = (
                        model.module if is_distributed else model
                    ).scaler(systems, predictions)
                    scaled_targets = (model.module if is_distributed else model).scaler(
                        systems, targets
                    )
                    train_rmse_calculator.update(
                        scaled_predictions, scaled_targets, extra_data
                    )
                    if self.hypers["log_mae"]:
                        train_mae_calculator.update(
                            scaled_predictions, scaled_targets, extra_data
                        )

            # Compute train metrics if they are to be logged this epoch:
            if epoch == start_epoch or epoch % self.hypers["log_interval"] == 0:
                finalized_train_info = train_rmse_calculator.finalize(
                    not_per_atom=["positions_gradients"] + per_structure_targets,
                    is_distributed=is_distributed,
                    device=device,
                )
                if self.hypers["log_mae"]:
                    finalized_train_info.update(
                        train_mae_calculator.finalize(
                            not_per_atom=["positions_gradients"]
                            + per_structure_targets,
                            is_distributed=is_distributed,
                            device=device,
                        )
                    )

            val_loss = 0.0
            for batch in val_dataloader:
                systems, targets, extra_data = unpack_batch(batch)
                systems, targets, extra_data = batch_to(
                    systems, targets, extra_data, device=device
                )
                for additive_model in raw_model.additive_models:
                    targets = remove_additive(
                        systems, targets, additive_model, train_targets
                    )
                targets = remove_scale(systems, targets, raw_model.scaler)
                systems, targets, extra_data = batch_to(
                    systems, targets, extra_data, dtype=dtype
                )

                predictions = evaluate_model(
                    model,
                    systems,
                    {key: train_targets[key] for key in targets.keys()},
                    is_training=False,
                )

                # average by the number of atoms
                predictions = average_by_num_atoms(
                    predictions, systems, per_structure_targets
                )
                targets = average_by_num_atoms(targets, systems, per_structure_targets)

                # Apply per-property scales to the predictions before loss computation.
                # The targets from the dataloader have only been scaled per-target, and
                # not per-property. This transformation only applies to targets with
                # per-property scales (i.e. multiple blocks or multiple properties), and
                # leaves the others unchanged.
                predictions = (model.module if is_distributed else model).scaler(
                    systems,
                    predictions,
                    remove=False,
                    use_per_target_scales=False,  # never before loss
                    use_per_property_scales=True,
                )

                val_loss_batch = loss_fn(predictions, targets, extra_data)

                if is_distributed:
                    # sum the loss over all processes
                    torch.distributed.all_reduce(val_loss_batch)
                val_loss += val_loss_batch.item()

                # Reapply scales and accumulate quantities for computing val
                # metrics. This is done for every epoch as validation metrics are
                # needed for model selection
                scaled_predictions = (model.module if is_distributed else model).scaler(
                    systems, predictions
                )
                scaled_targets = (model.module if is_distributed else model).scaler(
                    systems, targets
                )

                val_rmse_calculator.update(
                    scaled_predictions, scaled_targets, extra_data
                )
                if self.hypers["log_mae"]:
                    val_mae_calculator.update(
                        scaled_predictions, scaled_targets, extra_data
                    )

            # Compute val metrics:
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
            if epoch == start_epoch or epoch % self.hypers["log_interval"] == 0:
                finalized_train_info = {"loss": train_loss, **finalized_train_info}
            finalized_val_info = {"loss": val_loss, **finalized_val_info}

            if epoch == start_epoch:
                metric_logger = MetricLogger(
                    log_obj=ROOT_LOGGER,
                    dataset_info=raw_model.dataset_info,
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
                if new_lr < _MIN_LEARNING_RATE:
                    logging.info("Learning rate is too small, stopping training")
                    break
                else:
                    logging.info(f"Changing learning rate from {old_lr} to {new_lr}")
                    old_lr = new_lr
                    # load best model and optimizer state dict, re-initialize scheduler
                    raw_model.load_state_dict(self.best_model_state_dict)
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
                self.best_model_state_dict = copy.deepcopy(raw_model.state_dict())
                self.best_epoch = epoch
                self.best_optimizer_state_dict = copy.deepcopy(optimizer.state_dict())

            if epoch % self.hypers["checkpoint_interval"] == 0:
                if is_distributed:
                    torch.distributed.barrier()
                self.optimizer_state_dict = optimizer.state_dict()
                self.scheduler_state_dict = lr_scheduler.state_dict()
                self.epoch = epoch
                if rank == 0:
                    self.save_checkpoint(
                        raw_model,
                        Path(checkpoint_dir) / f"model_{epoch}.ckpt",
                    )

        # prepare for the checkpoint that will be saved outside the function
        self.epoch = epoch
        self.optimizer_state_dict = optimizer.state_dict()
        self.scheduler_state_dict = lr_scheduler.state_dict()
        checkpoint = raw_model.get_checkpoint()
        checkpoint.update(
            {
                "train_hypers": self.hypers,
                "trainer_ckpt_version": self.__checkpoint_version__,
                "epoch": self.epoch,
                "optimizer_state_dict": self.optimizer_state_dict,
                "scheduler_state_dict": self.scheduler_state_dict,
                "best_epoch": self.best_epoch,
                "best_metric": self.best_metric,
                "best_model_state_dict": self.best_model_state_dict,
                "best_optimizer_state_dict": self.best_optimizer_state_dict,
            }
        )

        if is_distributed:
            torch.distributed.destroy_process_group()

    def save_checkpoint(self, model, path: Union[str, Path]):
        checkpoint = model.get_checkpoint()
        checkpoint.update(
            {
                "train_hypers": self.hypers,
                "trainer_ckpt_version": self.__checkpoint_version__,
                "epoch": self.epoch,
                "optimizer_state_dict": self.optimizer_state_dict,
                "scheduler_state_dict": self.scheduler_state_dict,
                "best_epoch": self.best_epoch,
                "best_metric": self.best_metric,
                "best_model_state_dict": self.best_model_state_dict,
                "best_optimizer_state_dict": self.best_optimizer_state_dict,
            }
        )
        torch.save(
            checkpoint,
            check_file_extension(path, ".ckpt"),
        )

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint: Dict[str, Any],
        hypers: TrainerHypers,
        context: Literal["restart", "finetune"],
    ) -> "Trainer":
        trainer = cls(hypers)
        trainer.optimizer_state_dict = checkpoint["optimizer_state_dict"]
        trainer.scheduler_state_dict = checkpoint["scheduler_state_dict"]
        if context == "restart":
            trainer.epoch = checkpoint["epoch"]
        else:
            assert context == "finetune"
            trainer.epoch = None
        trainer.best_epoch = checkpoint["best_epoch"]
        trainer.best_metric = checkpoint["best_metric"]
        trainer.best_model_state_dict = checkpoint["best_model_state_dict"]
        trainer.best_optimizer_state_dict = checkpoint["best_optimizer_state_dict"]

        return trainer

    @classmethod
    def upgrade_checkpoint(cls, checkpoint: Dict) -> Dict:
        for v in range(1, cls.__checkpoint_version__):
            if checkpoint["trainer_ckpt_version"] == v:
                update = getattr(checkpoints, f"trainer_update_v{v}_v{v + 1}")
                update(checkpoint)
                checkpoint["trainer_ckpt_version"] = v + 1

        if checkpoint["trainer_ckpt_version"] != cls.__checkpoint_version__:
            raise RuntimeError(
                f"Unable to upgrade the checkpoint: the checkpoint is using trainer "
                f"version {checkpoint['trainer_ckpt_version']}, while the current "
                f"trainer version is {cls.__checkpoint_version__}."
            )
        return checkpoint
