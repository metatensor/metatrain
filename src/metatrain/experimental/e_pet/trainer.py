from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union, cast

import torch

from metatrain.pet.trainer import Trainer as PETTrainer
from metatrain.utils.data import Dataset
from metatrain.utils.data import unpack_batch
from metatrain.utils.data.atomic_basis_helpers import (
    get_prepare_atomic_basis_targets_transform,
)
from metatrain.utils.distributed.batch_utils import should_skip_batch
from metatrain.utils.evaluate_model import evaluate_model
from metatrain.utils.loss import LossAggregator, LossSpecification
from metatrain.utils.per_atom import average_by_num_atoms
from metatrain.utils.training_diagnostics import (
    assert_finite_loss,
    assert_finite_metrics,
)
from metatrain.utils.transfer import batch_to

from .documentation import TrainerHypers
from .model import EPET


class Trainer(PETTrainer):
    __checkpoint_version__ = 1

    def __init__(self, hypers: TrainerHypers) -> None:
        super().__init__(hypers)

    def _has_split_learning_rates(self) -> bool:
        return any(
            self.hypers.get(key) is not None
            for key in (
                "pet_trunk_learning_rate",
                "tensor_basis_learning_rate",
                "readout_learning_rate",
            )
        )

    def _build_optimizer(self, model: EPET) -> torch.optim.Optimizer:
        base_lr = float(self.hypers["learning_rate"])
        trunk_lr = float(self.hypers.get("pet_trunk_learning_rate") or base_lr)
        basis_lr = float(self.hypers.get("tensor_basis_learning_rate") or base_lr)
        readout_lr = float(self.hypers.get("readout_learning_rate") or base_lr)

        if not self._has_split_learning_rates():
            if self.hypers["weight_decay"] is not None:
                return torch.optim.AdamW(
                    model.parameters(),
                    lr=base_lr,
                    weight_decay=self.hypers["weight_decay"],
                )
            return torch.optim.Adam(model.parameters(), lr=base_lr)

        basis_prefixes = ("basis_calculators.",)
        readout_prefixes = (
            "node_heads.",
            "edge_heads.",
            "node_last_layers.",
            "edge_last_layers.",
        )
        trunk_params: List[torch.nn.Parameter] = []
        basis_params: List[torch.nn.Parameter] = []
        readout_params: List[torch.nn.Parameter] = []
        seen: set[int] = set()

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            param_id = id(param)
            if param_id in seen:
                continue
            seen.add(param_id)
            if name.startswith(basis_prefixes):
                basis_params.append(param)
            elif name.startswith(readout_prefixes):
                readout_params.append(param)
            else:
                trunk_params.append(param)

        param_groups: List[Dict[str, Any]] = []
        if trunk_params:
            param_groups.append(
                {"params": trunk_params, "lr": trunk_lr, "name": "pet_trunk"}
            )
        if basis_params:
            param_groups.append(
                {"params": basis_params, "lr": basis_lr, "name": "tensor_basis"}
            )
        if readout_params:
            param_groups.append(
                {"params": readout_params, "lr": readout_lr, "name": "readout"}
            )

        for group in param_groups:
            num_params = sum(param.numel() for param in group["params"])
            logging.info(
                "Optimizer group %s: lr=%s params=%s",
                group["name"],
                group["lr"],
                num_params,
            )

        if self.hypers["weight_decay"] is not None:
            return torch.optim.AdamW(
                param_groups,
                lr=base_lr,
                weight_decay=self.hypers["weight_decay"],
            )
        return torch.optim.Adam(param_groups, lr=base_lr)

    def _load_restart_state(
        self,
        model: EPET,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    ) -> None:
        if getattr(model, "has_new_targets", False):
            return
        if self.optimizer_state_dict is not None:
            optimizer.load_state_dict(self.optimizer_state_dict)
        if self.scheduler_state_dict is not None:
            lr_scheduler.load_state_dict(self.scheduler_state_dict)

    def train(
        self,
        model: EPET,
        dtype: torch.dtype,
        devices: List[torch.device],
        train_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        val_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        checkpoint_dir: str,
    ) -> None:
        coefficient_l2_weight = float(self.hypers.get("coefficient_l2_weight", 0.0))
        basis_gram_weight = float(self.hypers.get("basis_gram_weight", 0.0))
        if (
            coefficient_l2_weight == 0.0
            and basis_gram_weight == 0.0
            and not self._has_split_learning_rates()
        ):
            super().train(model, dtype, devices, train_datasets, val_datasets, checkpoint_dir)
            return

        assert dtype in EPET.__supported_dtypes__

        is_distributed = self.hypers["distributed"]
        is_finetune = self.hypers["finetune"]["read_from"] is not None
        if is_finetune:
            raise NotImplementedError(
                "Custom e-pet training with split learning rates or basis penalties "
                "does not support finetuning yet."
            )
        if is_distributed:
            raise NotImplementedError(
                "Custom e-pet training with split learning rates or basis penalties "
                "does not support distributed training yet."
            )

        device = devices[0]
        logging.info(f"Training on device {device} with dtype {dtype}")

        from metatrain.pet.trainer import get_scheduler
        from metatrain.utils.additive import get_remove_additive_transform
        from metatrain.utils.augmentation import RotationalAugmenter
        from metatrain.utils.data import (
            CollateFn,
            CombinedDataLoader,
            get_num_workers,
            validate_num_workers,
        )
        from metatrain.utils.logging import ROOT_LOGGER, MetricLogger
        from metatrain.utils.metrics import (
            MAEAccumulator,
            RMSEAccumulator,
            get_selected_metric,
        )
        from metatrain.utils.neighbor_lists import (
            get_requested_neighbor_lists,
            get_system_with_neighbor_lists_transform,
        )
        from metatrain.utils.scaler import get_remove_scale_transform
        from torch.utils.data import DataLoader

        model.to(device=device, dtype=dtype)
        for additive_model in model.additive_models:
            additive_model.to(dtype=torch.float64)
        model.scaler.to(dtype=torch.float64)

        logging.info("Calculating composition weights")
        model.additive_models[0].train_model(
            train_datasets,
            model.additive_models[1:],
            self.hypers["batch_size"],
            False,
            self.hypers["atomic_baseline"],
        )
        if self.hypers["scale_targets"]:
            logging.info("Calculating scaling weights")
            model.scaler.train_model(
                train_datasets,
                model.additive_models,
                self.hypers["batch_size"],
                False,
                self.hypers["fixed_scaling_weights"],
            )

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

        dataset_info = model.dataset_info
        train_targets = dataset_info.targets
        extra_data_info = dataset_info.extra_data
        rotational_augmenter = RotationalAugmenter(
            target_info_dict=train_targets,
            extra_data_info_dict=extra_data_info,
        )
        requested_neighbor_lists = get_requested_neighbor_lists(model)
        atomic_basis_transform, atomic_basis_reverse_transform = (
            get_prepare_atomic_basis_targets_transform(train_targets, extra_data_info)
        )

        collate_fn_train = CollateFn(
            target_keys=list(train_targets.keys()),
            callables=[
                get_remove_additive_transform(additive_models, train_targets),
                get_remove_scale_transform(scaler),
                atomic_basis_transform,
                rotational_augmenter.apply_random_augmentations,
                get_system_with_neighbor_lists_transform(requested_neighbor_lists),
            ],
            batch_atom_bounds=self.hypers["batch_atom_bounds"],
        )
        collate_fn_val = CollateFn(
            target_keys=list(train_targets.keys()),
            callables=[
                get_system_with_neighbor_lists_transform(requested_neighbor_lists),
                get_remove_additive_transform(additive_models, train_targets),
                get_remove_scale_transform(scaler),
                atomic_basis_transform,
            ],
            batch_atom_bounds=self.hypers["batch_atom_bounds"],
        )

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
        for train_dataset in train_datasets:
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
                    shuffle=True,
                    drop_last=True,
                    collate_fn=collate_fn_train,
                    num_workers=num_workers,
                )
            )
        train_dataloader = CombinedDataLoader(train_dataloaders, shuffle=True)

        val_dataloaders = []
        for val_dataset in val_datasets:
            val_dataloaders.append(
                DataLoader(
                    dataset=val_dataset,
                    batch_size=self.hypers["batch_size"],
                    shuffle=False,
                    drop_last=False,
                    collate_fn=collate_fn_val,
                    num_workers=num_workers,
                )
            )
        val_dataloader = CombinedDataLoader(val_dataloaders, shuffle=False)

        loss_hypers = cast(Dict[str, LossSpecification], self.hypers["loss"])
        loss_fn = LossAggregator(targets=train_targets, config=loss_hypers)
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

        optimizer = self._build_optimizer(model)
        lr_scheduler = get_scheduler(optimizer, self.hypers, len(train_dataloader))
        self._load_restart_state(model, optimizer, lr_scheduler)

        per_structure_targets = self.hypers["per_structure_targets"]
        logging.info(f"Base learning rate: {self.hypers['learning_rate']}")
        start_epoch = 0 if self.epoch is None else self.epoch + 1
        if self.best_metric is None:
            self.best_metric = float("inf")
        logging.info("Starting training")
        epoch = start_epoch

        metric_logger = None
        for epoch in range(start_epoch, self.hypers["num_epochs"]):
            train_rmse_calculator = RMSEAccumulator(self.hypers["log_separate_blocks"])
            val_rmse_calculator = RMSEAccumulator(self.hypers["log_separate_blocks"])
            if self.hypers["log_mae"]:
                train_mae_calculator = MAEAccumulator(
                    self.hypers["log_separate_blocks"]
                )
                val_mae_calculator = MAEAccumulator(self.hypers["log_separate_blocks"])

            train_loss = 0.0
            for batch in train_dataloader:
                if should_skip_batch(batch, False, device):
                    continue

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

                predictions = average_by_num_atoms(
                    predictions, systems, per_structure_targets
                )
                targets = average_by_num_atoms(targets, systems, per_structure_targets)
                train_loss_batch = loss_fn(predictions, targets, extra_data)

                if coefficient_l2_weight != 0.0:
                    train_loss_batch = (
                        train_loss_batch
                        + coefficient_l2_weight * model.get_regularization_loss()
                    )
                if basis_gram_weight != 0.0:
                    train_loss_batch = (
                        train_loss_batch
                        + basis_gram_weight * model.get_basis_gram_loss()
                    )
                assert_finite_loss(
                    train_loss_batch,
                    phase="training",
                    predictions=predictions,
                    targets=targets,
                    extra_data=extra_data,
                )

                train_loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.hypers["grad_clip_norm"]
                )
                optimizer.step()
                lr_scheduler.step()
                train_loss += train_loss_batch.item()

                systems, targets, extra_data = atomic_basis_reverse_transform(
                    systems, targets, extra_data
                )
                systems, predictions, _ = atomic_basis_reverse_transform(
                    systems, predictions, {}
                )
                scaled_predictions = model.scaler(systems, predictions)
                scaled_targets = model.scaler(systems, targets)
                try:
                    train_rmse_calculator.update(
                        scaled_predictions, scaled_targets, extra_data
                    )
                except ValueError as err:
                    raise ValueError(
                        f"Non-finite scaled training metric inputs: {err}"
                    ) from err
                if self.hypers["log_mae"]:
                    try:
                        train_mae_calculator.update(
                            scaled_predictions, scaled_targets, extra_data
                        )
                    except ValueError as err:
                        raise ValueError(
                            f"Non-finite scaled training metric inputs: {err}"
                        ) from err

            finalized_train_info = train_rmse_calculator.finalize(
                not_per_atom=["positions_gradients"] + per_structure_targets,
                is_distributed=False,
                device=device,
            )
            if self.hypers["log_mae"]:
                finalized_train_info.update(
                    train_mae_calculator.finalize(
                        not_per_atom=["positions_gradients"] + per_structure_targets,
                        is_distributed=False,
                        device=device,
                    )
                )

            val_loss = 0.0
            for batch in val_dataloader:
                if should_skip_batch(batch, False, device):
                    continue

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

                predictions = average_by_num_atoms(
                    predictions, systems, per_structure_targets
                )
                targets = average_by_num_atoms(targets, systems, per_structure_targets)
                val_loss_batch = loss_fn(predictions, targets, extra_data)
                assert_finite_loss(
                    val_loss_batch,
                    phase="validation",
                    predictions=predictions,
                    targets=targets,
                    extra_data=extra_data,
                )
                val_loss += val_loss_batch.item()

                systems, targets, extra_data = atomic_basis_reverse_transform(
                    systems, targets, extra_data
                )
                systems, predictions, _ = atomic_basis_reverse_transform(
                    systems, predictions, {}
                )
                scaled_predictions = model.scaler(systems, predictions)
                scaled_targets = model.scaler(systems, targets)
                try:
                    val_rmse_calculator.update(
                        scaled_predictions, scaled_targets, extra_data
                    )
                except ValueError as err:
                    raise ValueError(
                        f"Non-finite scaled validation metric inputs: {err}"
                    ) from err
                if self.hypers["log_mae"]:
                    try:
                        val_mae_calculator.update(
                            scaled_predictions, scaled_targets, extra_data
                        )
                    except ValueError as err:
                        raise ValueError(
                            f"Non-finite scaled validation metric inputs: {err}"
                        ) from err

            finalized_val_info = val_rmse_calculator.finalize(
                not_per_atom=["positions_gradients"] + per_structure_targets,
                is_distributed=False,
                device=device,
            )
            if self.hypers["log_mae"]:
                finalized_val_info.update(
                    val_mae_calculator.finalize(
                        not_per_atom=["positions_gradients"] + per_structure_targets,
                        is_distributed=False,
                        device=device,
                    )
                )

            finalized_train_info = {"loss": train_loss, **finalized_train_info}
            finalized_val_info = {"loss": val_loss, **finalized_val_info}
            assert_finite_metrics(finalized_train_info, phase="training")
            assert_finite_metrics(finalized_val_info, phase="validation")

            if epoch == start_epoch:
                metric_logger = MetricLogger(
                    log_obj=ROOT_LOGGER,
                    dataset_info=model.dataset_info,
                    initial_metrics=[finalized_train_info, finalized_val_info],
                    names=["training", "validation"],
                )
            if epoch % self.hypers["log_interval"] == 0:
                metric_logger.log(
                    metrics=[finalized_train_info, finalized_val_info],
                    epoch=epoch,
                    rank=0,
                    learning_rate=optimizer.param_groups[0]["lr"],
                )

            val_metric = get_selected_metric(
                finalized_val_info, self.hypers["best_model_metric"]
            )
            if val_metric < self.best_metric:
                self.best_metric = val_metric
                self.best_model_state_dict = copy.deepcopy(model.state_dict())
                self.best_epoch = epoch
                self.best_optimizer_state_dict = copy.deepcopy(optimizer.state_dict())

            if epoch % self.hypers["checkpoint_interval"] == 0:
                self.optimizer_state_dict = optimizer.state_dict()
                self.scheduler_state_dict = lr_scheduler.state_dict()
                self.epoch = epoch
                self.save_checkpoint(model, Path(checkpoint_dir) / f"model_{epoch}.ckpt")

        self.epoch = epoch
        self.optimizer_state_dict = optimizer.state_dict()
        self.scheduler_state_dict = lr_scheduler.state_dict()

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
            trainer.epoch = None
        trainer.best_epoch = checkpoint["best_epoch"]
        trainer.best_metric = checkpoint["best_metric"]
        trainer.best_model_state_dict = checkpoint["best_model_state_dict"]
        trainer.best_optimizer_state_dict = checkpoint["best_optimizer_state_dict"]
        return trainer

    @classmethod
    def upgrade_checkpoint(cls, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        if checkpoint["trainer_ckpt_version"] != cls.__checkpoint_version__:
            raise RuntimeError(
                "Unable to upgrade the checkpoint: the checkpoint is using trainer "
                f"version {checkpoint['trainer_ckpt_version']}, while the current "
                f"trainer version is {cls.__checkpoint_version__}."
            )
        return checkpoint
