import argparse
import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import torch
from mace.tools.scripts_utils import (
    LRScheduler,
    get_optimizer,
    get_params_options,
)
from torch.utils.data import DataLoader, DistributedSampler

from metatrain.utils.abc import ModelInterface, TrainerInterface
from metatrain.utils.additive import get_remove_additive_transform
from metatrain.utils.data import (
    CollateFn,
    CombinedDataLoader,
    Dataset,
    get_num_workers,
    unpack_batch,
    validate_num_workers,
)
from metatrain.utils.distributed.distributed_data_parallel import (
    DistributedDataParallel,
)
from metatrain.utils.distributed.slurm import DistributedEnvironment
from metatrain.utils.evaluate_model import evaluate_model
from metatrain.utils.io import check_file_extension
from metatrain.utils.logging import ROOT_LOGGER, MetricLogger
from metatrain.utils.loss import LossAggregator
from metatrain.utils.metrics import MAEAccumulator, RMSEAccumulator, get_selected_metric
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists_transform,
)
from metatrain.utils.per_atom import average_by_num_atoms
from metatrain.utils.scaler import get_remove_scale_transform
from metatrain.utils.transfer import batch_to

from . import checkpoints
from .model import MetaMACE
from .modules.finetuning import apply_finetuning_strategy


def get_optimizer_and_scheduler(
    trainer_hypers: dict[str, Any],
    model: MetaMACE,
    optimizer_state_dict: Optional[dict[str, Any]] = None,
    scheduler_state_dict: Optional[dict[str, Any]] = None,
    is_distributed: bool = False,
) -> tuple[torch.optim.Optimizer, LRScheduler]:
    """Initialize the optimizer and scheduler as implemented in MACE.

    It just uses the functions from mace.tools.scripts_utils to create them.

    :param trainer_hypers: The trainer hyperparameters as provided in metatrain.yaml
    :param model: The MetaMACE model to optimize.
    :param optimizer_state_dict: The state dict of the optimizer to resume from, if any.
    :param scheduler_state_dict: The state dict of the scheduler to resume from, if any.
    :param is_distributed: Whether the training is distributed over GPUs or not.

    :return: A tuple with the optimizer and the scheduler.
    """
    opt_args = argparse.Namespace(
        lr=trainer_hypers["learning_rate"],
        weight_decay=trainer_hypers["weight_decay"],
        amsgrad=trainer_hypers["amsgrad"],
        beta=trainer_hypers["beta"],
        optimizer=trainer_hypers["optimizer"],
        scheduler=trainer_hypers["lr_scheduler"],
        lr_scheduler_gamma=trainer_hypers["lr_scheduler_gamma"],
        lr_factor=trainer_hypers["lr_factor"],
        scheduler_patience=trainer_hypers["lr_scheduler_patience"],
    )

    opt_options = get_params_options(opt_args, model.mace_model)

    # Add heads, additive models and scaler parameters to the optimizer. Although the
    # additive models and scaler weights are not optimized, this maintains consistency
    # with PET, where all model parameters (including the additive models stored as
    # attributes) are passed to the optimizer.
    opt_options["params"].extend(
        [
            {"name": "heads", "params": model.heads.parameters()},
            {"name": "additive_models", "params": model.additive_models.parameters()},
            {"name": "scaler", "params": model.scaler.parameters()},
        ]
    )

    optimizer = get_optimizer(opt_args, opt_options)

    is_finetune = "finetune" in trainer_hypers

    if optimizer_state_dict is not None and not is_finetune:
        # try to load the optimizer state dict, but this is only possible
        # if there are no new targets in the model (new parameters)
        if not (model.module if is_distributed else model).has_new_targets:
            optimizer.load_state_dict(optimizer_state_dict)

    scheduler = LRScheduler(optimizer, opt_args)

    if scheduler_state_dict is not None and not is_finetune:
        # same as the optimizer, try to load the scheduler state dict
        if not (model.module if is_distributed else model).has_new_targets:
            scheduler.load_state_dict(scheduler_state_dict)

    return optimizer, scheduler


class Trainer(TrainerInterface):
    __checkpoint_version__ = 1

    def __init__(self, hypers: Dict[str, Any]) -> None:
        super().__init__(hypers)

        self.optimizer_state_dict: Optional[Dict[str, Any]] = None
        self.scheduler_state_dict: Optional[Dict[str, Any]] = None
        self.epoch: Optional[int] = None
        self.best_epoch: Optional[int] = None
        self.best_metric: Optional[float] = None
        self.best_model_state_dict: Optional[Dict[str, Any]] = None
        self.best_optimizer_state_dict: Optional[Dict[str, Any]] = None

    def train(
        self,
        model: MetaMACE,
        dtype: torch.dtype,
        devices: List[torch.device],
        train_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        val_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        checkpoint_dir: str,
    ) -> None:
        assert dtype in MetaMACE.__supported_dtypes__

        is_distributed = self.hypers["distributed"]
        is_finetune = "finetune" in self.hypers

        if is_distributed:
            if len(devices) > 1:
                raise ValueError(
                    "Requested distributed training with the `multi-gpu` device. "
                    " If you want to run distributed training with MetaMACE, please "
                    "set `device` to cuda."
                )
            # the calculation of the device number works both when GPUs on different
            # processes are not visible to each other and when they are
            distr_env = DistributedEnvironment(self.hypers["distributed_port"])
            device_number = distr_env.local_rank % torch.cuda.device_count()
            device = torch.device("cuda", device_number)
            torch.distributed.init_process_group(backend="nccl", device_id=device)
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
        else:
            rank = 0
            device = devices[0]
            # only one device, as we don't support non-distributed multi-gpu for now

        if is_distributed:
            logging.info(f"Training on {world_size} devices with dtype {dtype}")
        else:
            logging.info(f"Training on device {device} with dtype {dtype}")

        # Apply fine-tuning strategy if provided
        if is_finetune:
            model = apply_finetuning_strategy(model, self.hypers["finetune"])
            method = self.hypers["finetune"]["method"]
            num_params = sum(p.numel() for p in model.parameters())
            num_trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )

            logging.info(f"Applied finetuning strategy: {method}")
            logging.info(
                f"Number of trainable parameters: {num_trainable_params} "
                f"[{num_trainable_params / num_params:.2%} %]"
            )

        # Move the model to the device and dtype:
        model.to(device=device, dtype=dtype)
        # The additive models of MetaMACE are always in float64 (to avoid numerical
        # errors in the composition weights, which can be very large).
        for additive_model in model.additive_models:
            additive_model.to(dtype=torch.float64)
        model.scaler.to(dtype=torch.float64)

        logging.info("Calculating composition weights")

        model.additive_models[0].train_model(  # this is the composition model
            train_datasets,
            model.additive_models[1:],
            self.hypers["batch_size"],
            is_distributed,
            self.hypers["fixed_composition_weights"],
        )

        if self.hypers["scale_targets"]:
            logging.info("Calculating scaling weights")
            model.scaler.train_model(
                train_datasets,
                model.additive_models,
                self.hypers["batch_size"],
                is_distributed,
                self.hypers["fixed_scaling_weights"],
            )

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

        # Extract additive models and scaler and move them to CPU/float64 so they
        # can be used in the collate function
        model.additive_models[0].weights_to(device="cpu", dtype=torch.float64)
        additive_models = copy.deepcopy(
            model.additive_models.to(dtype=torch.float64, device="cpu")
        )
        model.additive_models.to(device)
        model.additive_models[0].weights_to(device=device, dtype=torch.float64)
        model.scaler.scales_to(device="cpu", dtype=torch.float64)
        scaler = copy.deepcopy(model.scaler.to(dtype=torch.float64, device="cpu"))
        model.scaler.to(device)
        model.scaler.scales_to(device=device, dtype=torch.float64)

        # Create collate functions:
        dataset_info = model.dataset_info
        train_targets = dataset_info.targets
        requested_neighbor_lists = get_requested_neighbor_lists(model)
        collate_fn_train = CollateFn(
            target_keys=list(train_targets.keys()),
            callables=[
                get_system_with_neighbor_lists_transform(requested_neighbor_lists),
                get_remove_additive_transform(additive_models, train_targets),
                get_remove_scale_transform(scaler),
            ],
        )
        collate_fn_val = CollateFn(
            target_keys=list(train_targets.keys()),
            callables=[  # no augmentation for validation
                get_system_with_neighbor_lists_transform(requested_neighbor_lists),
                get_remove_additive_transform(additive_models, train_targets),
                get_remove_scale_transform(scaler),
            ],
        )

        # Create dataloader for the training datasets:
        if self.hypers["num_workers"] is None:
            num_workers = get_num_workers()
            logging.info(
                "Number of workers for data-loading not provided and chosen "
                f"automatically. Using {num_workers} workers."
            )
        else:
            num_workers = self.hypers["num_workers"]
            validate_num_workers(num_workers)

        train_dataloaders = []
        for train_dataset, train_sampler in zip(
            train_datasets, train_samplers, strict=True
        ):
            if len(train_dataset) < self.hypers["batch_size"]:
                raise ValueError(
                    f"A training dataset has fewer samples "
                    f"({len(train_dataset)}) than the batch size "
                    f"({self.hypers['batch_size']}). "
                    "Please reduce the batch size."
                )
            train_dataloaders.append(
                DataLoader(
                    dataset=train_dataset,
                    batch_size=self.hypers["batch_size"],
                    sampler=train_sampler,
                    shuffle=(
                        # the sampler takes care of this (if present)
                        train_sampler is None
                    ),
                    drop_last=(
                        # the sampler takes care of this (if present)
                        train_sampler is None
                    ),
                    collate_fn=collate_fn_train,
                    num_workers=num_workers,
                )
            )
        train_dataloader = CombinedDataLoader(train_dataloaders, shuffle=True)

        # Create dataloader for the validation datasets:
        val_dataloaders = []
        for val_dataset, val_sampler in zip(val_datasets, val_samplers, strict=True):
            if len(val_dataset) < self.hypers["batch_size"]:
                raise ValueError(
                    f"A validation dataset has fewer samples "
                    f"({len(val_dataset)}) than the batch size "
                    f"({self.hypers['batch_size']}). "
                    "Please reduce the batch size."
                )
            val_dataloaders.append(
                DataLoader(
                    dataset=val_dataset,
                    batch_size=self.hypers["batch_size"],
                    sampler=val_sampler,
                    shuffle=False,
                    drop_last=False,
                    collate_fn=collate_fn_val,
                    num_workers=num_workers,
                )
            )
        val_dataloader = CombinedDataLoader(val_dataloaders, shuffle=False)

        if is_distributed:
            model = DistributedDataParallel(model, device_ids=[device])

        outputs_list = []
        for target_name, target_info in train_targets.items():
            outputs_list.append(target_name)
            for gradient_name in target_info.gradients:
                outputs_list.append(f"{target_name}_{gradient_name}_gradients")

        # Create a loss function:
        loss_hypers = self.hypers["loss"]
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

        optimizer, lr_scheduler = get_optimizer_and_scheduler(
            self.hypers,
            model,
            self.optimizer_state_dict,
            self.scheduler_state_dict,
            is_distributed,
        )

        per_structure_targets = self.hypers["per_structure_targets"]

        # Log the initial learning rate:
        logging.info(f"Base learning rate: {self.hypers['learning_rate']}")

        start_epoch = 0 if self.epoch is None else self.epoch + 1

        # Train the model:
        if self.best_metric is None:
            self.best_metric = float("inf")
        logging.info("Starting training")
        epoch = start_epoch

        for epoch in range(start_epoch, self.hypers["num_epochs"]):
            if is_distributed:
                for train_sampler in train_samplers:
                    train_sampler.set_epoch(epoch)
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
                    systems, targets, extra_data, dtype=dtype, device=device
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
                train_loss_batch = loss_fn(predictions, targets, extra_data)
                train_loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.hypers["grad_clip_norm"]
                )
                optimizer.step()

                if is_distributed:
                    # sum the loss over all processes
                    torch.distributed.all_reduce(train_loss_batch)
                train_loss += train_loss_batch.item()

                scaled_predictions = (model.module if is_distributed else model).scaler(
                    systems, predictions
                )
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
            for batch in val_dataloader:
                systems, targets, extra_data = unpack_batch(batch)
                systems, targets, extra_data = batch_to(
                    systems, targets, extra_data, dtype=dtype, device=device
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
                val_loss_batch = loss_fn(predictions, targets, extra_data)

                if is_distributed:
                    # sum the loss over all processes
                    torch.distributed.all_reduce(val_loss_batch)
                val_loss += val_loss_batch.item()

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

            lr_scheduler.step(metrics=val_loss)

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
            finalized_train_info = {
                "loss": train_loss,
                **finalized_train_info,
            }
            finalized_val_info = {
                "loss": val_loss,
                **finalized_val_info,
            }

            if epoch == start_epoch:
                metric_logger = MetricLogger(
                    log_obj=ROOT_LOGGER,
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
                    learning_rate=optimizer.param_groups[0]["lr"],
                )

            val_metric = get_selected_metric(
                finalized_val_info, self.hypers["best_model_metric"]
            )
            if val_metric < self.best_metric:
                self.best_metric = val_metric
                self.best_model_state_dict = copy.deepcopy(
                    (model.module if is_distributed else model).state_dict()
                )
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
                        (model.module if is_distributed else model),
                        Path(checkpoint_dir) / f"model_{epoch}.ckpt",
                    )

        # prepare for the checkpoint that will be saved outside the function
        self.epoch = epoch
        self.optimizer_state_dict = optimizer.state_dict()
        self.scheduler_state_dict = lr_scheduler.state_dict()

        if is_distributed:
            torch.distributed.destroy_process_group()

    def save_checkpoint(self, model: ModelInterface, path: Union[str, Path]) -> None:
        checkpoint = model.get_checkpoint()
        if self.best_model_state_dict is not None:
            self.best_model_state_dict["finetune_config"] = model.finetune_config
        checkpoint.update(
            {
                "trainer_ckpt_version": self.__checkpoint_version__,
                "train_hypers": self.hypers,
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
        hypers: Dict[str, Any],
        context: Literal["restart", "finetune"],
    ) -> "Trainer":
        trainer = cls(hypers)
        trainer.optimizer_state_dict = checkpoint["optimizer_state_dict"]
        trainer.scheduler_state_dict = checkpoint["scheduler_state_dict"]
        if context == "restart":
            trainer.epoch = checkpoint["epoch"]
        else:
            assert "context" == "finetune"
            trainer.epoch = None  # interpreted as zero in training loop
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
                f"Unable to upgrade the checkpoint: the checkpoint is using "
                f"trainer version {checkpoint['trainer_ckpt_version']}, while the "
                f"current trainer version is {cls.__checkpoint_version__}."
            )
        return checkpoint
