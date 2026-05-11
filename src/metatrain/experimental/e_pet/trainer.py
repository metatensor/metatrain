from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union, cast

import torch
from metatensor.torch import TensorMap
from torch.utils.data import DataLoader

from metatrain.pet.trainer import (
    Trainer as PETTrainer,
)
from metatrain.pet.trainer import (
    _apply_finetuning_if_requested,
    _build_loss_objects,
    _fit_and_copy_preprocessors_for_training,
    _log_loss_metadata,
    _move_model_to_training_device,
    _resolve_num_workers,
    get_scheduler,
)
from metatrain.utils._atomic_basis_irrep_balanced_loss import (
    _AtomicBasisIrrepBalancedLoss,
)
from metatrain.utils.additive import get_remove_additive_transform
from metatrain.utils.augmentation import RotationalAugmenter
from metatrain.utils.data import (
    CollateFn,
    CombinedDataLoader,
    Dataset,
    unpack_batch,
)
from metatrain.utils.data.atomic_basis_helpers import (
    get_prepare_atomic_basis_targets_transform,
)
from metatrain.utils.distributed.batch_utils import should_skip_batch
from metatrain.utils.evaluate_model import evaluate_model
from metatrain.utils.logging import ROOT_LOGGER, MetricLogger
from metatrain.utils.loss import LossAggregator, LossSpecification
from metatrain.utils.metrics import MAEAccumulator, RMSEAccumulator, get_selected_metric
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists_transform,
)
from metatrain.utils.per_atom import average_by_num_atoms
from metatrain.utils.scaler import get_remove_scale_transform
from metatrain.utils.training_diagnostics import (
    assert_finite_loss,
    assert_finite_metrics,
)
from metatrain.utils.transfer import batch_to

from .documentation import TrainerHypers
from .model import EPET


class Trainer(PETTrainer):
    __checkpoint_version__ = 1

    @staticmethod
    def _has_trainable_tensor_basis_parameters(model: EPET) -> bool:
        return any(
            name.startswith("basis_calculators.") and parameter.requires_grad
            for name, parameter in model.named_parameters()
        )

    def _has_tensor_basis_learning_rate_override(
        self, model: Optional[EPET] = None
    ) -> bool:
        basis_lr = self.hypers.get("tensor_basis_learning_rate")
        if basis_lr is None or float(basis_lr) == float(self.hypers["learning_rate"]):
            return False
        if model is None:
            return True
        return self._has_trainable_tensor_basis_parameters(model)

    def _regularization_weights(self) -> tuple[float, float]:
        return (
            float(self.hypers.get("coefficient_l2_weight", 0.0)),
            float(self.hypers.get("basis_gram_weight", 0.0)),
        )

    def _exclude_spherical_l0_from_coefficient_l2(self) -> bool:
        return bool(self.hypers.get("coefficient_l2_exclude_spherical_l0", False))

    def _atomic_basis_irrep_balanced_loss_config(self) -> Dict[str, Dict[str, Any]]:
        return dict(self.hypers.get("atomic_basis_irrep_balanced_loss") or {})

    def _requires_custom_training_path(self, model: Optional[EPET] = None) -> bool:
        coefficient_l2_weight, basis_gram_weight = self._regularization_weights()
        return (
            coefficient_l2_weight != 0.0
            or basis_gram_weight != 0.0
            or self._has_tensor_basis_learning_rate_override(model)
            or bool(self._atomic_basis_irrep_balanced_loss_config())
        )

    def _validate_custom_training_path(self, dtype: torch.dtype) -> None:
        if dtype not in EPET.__supported_dtypes__:
            raise ValueError(
                f"Unsupported dtype for e-pet training: {dtype}. Supported dtypes "
                f"are: {sorted(str(item) for item in EPET.__supported_dtypes__)}."
            )

        if self.hypers["distributed"]:
            raise NotImplementedError(
                "The custom e-pet training path does not support distributed "
                "training yet."
            )
        if self.hypers["max_atoms_per_batch"] is not None:
            raise NotImplementedError(
                "The custom e-pet training path does not support "
                "max_atoms_per_batch yet. Use fixed batch_size batching or disable "
                "the E-PET options that require the custom trainer."
            )

    def _add_regularization(
        self,
        model: EPET,
        loss: torch.Tensor,
        coefficient_l2_weight: float,
        basis_gram_weight: float,
    ) -> torch.Tensor:
        if coefficient_l2_weight != 0.0:
            loss = loss + coefficient_l2_weight * model.get_regularization_loss(
                exclude_spherical_l0=self._exclude_spherical_l0_from_coefficient_l2()
            )
        if basis_gram_weight != 0.0:
            loss = loss + basis_gram_weight * model.get_basis_gram_loss()
        return loss

    def _build_optimizer(self, model: EPET) -> torch.optim.Optimizer:
        base_lr = float(self.hypers["learning_rate"])
        basis_lr = self.hypers.get("tensor_basis_learning_rate")
        if basis_lr is not None:
            basis_lr = float(basis_lr)

        if not self._has_tensor_basis_learning_rate_override(model):
            if self.hypers["weight_decay"] is not None:
                return torch.optim.AdamW(
                    model.parameters(),
                    lr=base_lr,
                    weight_decay=self.hypers["weight_decay"],
                )
            return torch.optim.Adam(model.parameters(), lr=base_lr)

        basis_prefixes = ("basis_calculators.",)
        pet_and_readout_params: List[torch.nn.Parameter] = []
        basis_params: List[torch.nn.Parameter] = []
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
            else:
                pet_and_readout_params.append(param)

        param_groups: List[Dict[str, Any]] = []
        if pet_and_readout_params:
            param_groups.append(
                {
                    "params": pet_and_readout_params,
                    "lr": base_lr,
                    "name": "pet_and_readout",
                }
            )
        if basis_params:
            param_groups.append(
                {"params": basis_params, "lr": basis_lr, "name": "tensor_basis"}
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

    @staticmethod
    def _zero_loss_like(predictions: Dict[str, TensorMap]) -> torch.Tensor:
        first_tensor_map = next(iter(predictions.values()))
        first_block = first_tensor_map.block(first_tensor_map.keys[0])
        return torch.zeros(
            (), dtype=first_block.values.dtype, device=first_block.values.device
        )

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

    def _build_custom_dataloaders(
        self,
        model: EPET,
        device: torch.device,
        train_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        val_datasets: List[Union[Dataset, torch.utils.data.Subset]],
    ) -> tuple[Any, CombinedDataLoader, CombinedDataLoader]:
        train_targets = model.dataset_info.targets
        extra_data_info = model.dataset_info.extra_data
        rotational_augmenter = RotationalAugmenter(
            target_info_dict=train_targets,
            extra_data_info_dict=extra_data_info,
        )
        requested_neighbor_lists = get_requested_neighbor_lists(model)
        atomic_basis_transform, atomic_basis_reverse_transform = (
            get_prepare_atomic_basis_targets_transform(train_targets, extra_data_info)
        )

        additive_models, scaler = _fit_and_copy_preprocessors_for_training(
            model,
            train_datasets,
            self.hypers,
            atomic_basis_transform,
            False,
            device,
        )

        collate_fn_train = CollateFn(
            target_keys=list(train_targets.keys()),
            callables=[
                atomic_basis_transform,
                rotational_augmenter.apply_random_augmentations,
                get_system_with_neighbor_lists_transform(requested_neighbor_lists),
                get_remove_additive_transform(additive_models, train_targets),
                get_remove_scale_transform(scaler),
            ],
            batch_atom_bounds=self.hypers["batch_atom_bounds"],
        )
        collate_fn_val = CollateFn(
            target_keys=list(train_targets.keys()),
            callables=[
                atomic_basis_transform,
                get_system_with_neighbor_lists_transform(requested_neighbor_lists),
                get_remove_additive_transform(additive_models, train_targets),
                get_remove_scale_transform(scaler),
            ],
            batch_atom_bounds=self.hypers["batch_atom_bounds"],
        )

        num_workers = _resolve_num_workers(self.hypers)

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
        return atomic_basis_reverse_transform, train_dataloader, val_dataloader

    def _build_custom_losses(
        self,
        model: EPET,
    ) -> tuple[
        Dict[str, Dict[str, Any]],
        Optional[_AtomicBasisIrrepBalancedLoss],
        Dict[str, Any],
        Optional[LossAggregator],
    ]:
        train_targets = model.dataset_info.targets
        custom_loss_config = self._atomic_basis_irrep_balanced_loss_config()
        loss_hypers = cast(Dict[str, LossSpecification], self.hypers["loss"])
        custom_loss_fn, normal_train_targets, loss_fn = _build_loss_objects(
            train_targets,
            loss_hypers,
            custom_loss_config,
            model.scaler,
            self.hypers["scale_targets"],
        )
        return custom_loss_config, custom_loss_fn, normal_train_targets, loss_fn

    def _compute_custom_loss_batch(
        self,
        systems: List[Any],
        predictions: Dict[str, TensorMap],
        targets: Dict[str, TensorMap],
        extra_data: Dict[str, TensorMap],
        normal_train_targets: Dict[str, Any],
        loss_fn: Optional[LossAggregator],
        custom_loss_fn: Optional[_AtomicBasisIrrepBalancedLoss],
        atomic_basis_reverse_transform: Any,
        model: EPET,
    ) -> torch.Tensor:
        loss = self._zero_loss_like(predictions)
        if loss_fn is not None:
            normal_predictions = {
                name: predictions[name]
                for name in normal_train_targets
                if name in predictions
            }
            if normal_predictions:
                normal_targets = {name: targets[name] for name in normal_predictions}
                loss = loss + loss_fn(normal_predictions, normal_targets, extra_data)
        if custom_loss_fn is not None:
            loss = loss + custom_loss_fn.compute(
                systems,
                predictions,
                targets,
                atomic_basis_reverse_transform,
                model.scaler,
            )
        return loss

    def _scaled_metric_inputs(
        self,
        systems: List[Any],
        predictions: Dict[str, TensorMap],
        targets: Dict[str, TensorMap],
        extra_data: Dict[str, TensorMap],
        atomic_basis_reverse_transform: Any,
        model: EPET,
    ) -> tuple[Dict[str, TensorMap], Dict[str, TensorMap], Dict[str, TensorMap]]:
        scaled_predictions = model.scaler(systems, predictions)
        scaled_targets = model.scaler(systems, targets)
        if self.hypers["log_separate_blocks"]:
            systems, scaled_targets, extra_data = atomic_basis_reverse_transform(
                systems, scaled_targets, extra_data
            )
            systems, scaled_predictions, _ = atomic_basis_reverse_transform(
                systems, scaled_predictions, {}
            )
        return scaled_predictions, scaled_targets, extra_data

    def _update_metric_calculators(
        self,
        rmse_calculator: RMSEAccumulator,
        mae_calculator: Optional[MAEAccumulator],
        scaled_predictions: Dict[str, TensorMap],
        scaled_targets: Dict[str, TensorMap],
        extra_data: Dict[str, TensorMap],
        phase: str,
    ) -> None:
        try:
            rmse_calculator.update(scaled_predictions, scaled_targets, extra_data)
        except ValueError as err:
            raise ValueError(f"Non-finite scaled {phase} metric inputs: {err}") from err
        if mae_calculator is not None:
            try:
                mae_calculator.update(scaled_predictions, scaled_targets, extra_data)
            except ValueError as err:
                raise ValueError(
                    f"Non-finite scaled {phase} metric inputs: {err}"
                ) from err

    @staticmethod
    def _finalize_metric_info(
        rmse_calculator: RMSEAccumulator,
        mae_calculator: Optional[MAEAccumulator],
        per_structure_targets: List[str],
        device: torch.device,
    ) -> Dict[str, float]:
        info = rmse_calculator.finalize(
            not_per_atom=["positions_gradients"] + per_structure_targets,
            is_distributed=False,
            device=device,
        )
        if mae_calculator is not None:
            info.update(
                mae_calculator.finalize(
                    not_per_atom=["positions_gradients"] + per_structure_targets,
                    is_distributed=False,
                    device=device,
                )
            )
        return info

    def _update_best_model(
        self,
        model: EPET,
        optimizer: torch.optim.Optimizer,
        finalized_val_info: Dict[str, float],
        epoch: int,
    ) -> None:
        val_metric = get_selected_metric(
            finalized_val_info, self.hypers["best_model_metric"]
        )
        if self.best_metric is None or val_metric < self.best_metric:
            self.best_metric = val_metric
            self.best_model_state_dict = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            self.best_optimizer_state_dict = copy.deepcopy(optimizer.state_dict())

    def _save_epoch_checkpoint(
        self,
        model: EPET,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        checkpoint_dir: str,
        epoch: int,
    ) -> None:
        if epoch % self.hypers["checkpoint_interval"] != 0:
            return
        self.optimizer_state_dict = optimizer.state_dict()
        self.scheduler_state_dict = lr_scheduler.state_dict()
        self.epoch = epoch
        self.save_checkpoint(model, Path(checkpoint_dir) / f"model_{epoch}.ckpt")

    def _new_metric_calculators(
        self,
    ) -> tuple[RMSEAccumulator, Optional[MAEAccumulator]]:
        rmse_calculator = RMSEAccumulator(self.hypers["log_separate_blocks"])
        mae_calculator = None
        if self.hypers["log_mae"]:
            mae_calculator = MAEAccumulator(self.hypers["log_separate_blocks"])
        return rmse_calculator, mae_calculator

    def _run_training_epoch(
        self,
        model: EPET,
        train_dataloader: CombinedDataLoader,
        train_targets: Dict[str, Any],
        normal_train_targets: Dict[str, Any],
        loss_fn: Optional[LossAggregator],
        custom_loss_fn: Optional[_AtomicBasisIrrepBalancedLoss],
        atomic_basis_reverse_transform: Any,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        coefficient_l2_weight: float,
        basis_gram_weight: float,
        per_structure_targets: List[str],
        dtype: torch.dtype,
        device: torch.device,
    ) -> Dict[str, float]:
        rmse_calculator, mae_calculator = self._new_metric_calculators()
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
            train_loss_batch = self._compute_custom_loss_batch(
                systems,
                predictions,
                targets,
                extra_data,
                normal_train_targets,
                loss_fn,
                custom_loss_fn,
                atomic_basis_reverse_transform,
                model,
            )
            train_loss_batch = self._add_regularization(
                model,
                train_loss_batch,
                coefficient_l2_weight,
                basis_gram_weight,
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

            scaled_predictions, scaled_targets, metric_extra_data = (
                self._scaled_metric_inputs(
                    systems,
                    predictions,
                    targets,
                    extra_data,
                    atomic_basis_reverse_transform,
                    model,
                )
            )
            self._update_metric_calculators(
                rmse_calculator,
                mae_calculator,
                scaled_predictions,
                scaled_targets,
                metric_extra_data,
                "training",
            )

        return {
            "loss": train_loss,
            **self._finalize_metric_info(
                rmse_calculator,
                mae_calculator,
                per_structure_targets,
                device,
            ),
        }

    def _run_validation_epoch(
        self,
        model: EPET,
        val_dataloader: CombinedDataLoader,
        train_targets: Dict[str, Any],
        normal_train_targets: Dict[str, Any],
        loss_fn: Optional[LossAggregator],
        custom_loss_fn: Optional[_AtomicBasisIrrepBalancedLoss],
        atomic_basis_reverse_transform: Any,
        per_structure_targets: List[str],
        dtype: torch.dtype,
        device: torch.device,
    ) -> Dict[str, float]:
        rmse_calculator, mae_calculator = self._new_metric_calculators()
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
            val_loss_batch = self._compute_custom_loss_batch(
                systems,
                predictions,
                targets,
                extra_data,
                normal_train_targets,
                loss_fn,
                custom_loss_fn,
                atomic_basis_reverse_transform,
                model,
            )
            assert_finite_loss(
                val_loss_batch,
                phase="validation",
                predictions=predictions,
                targets=targets,
                extra_data=extra_data,
            )
            val_loss += val_loss_batch.item()

            scaled_predictions, scaled_targets, metric_extra_data = (
                self._scaled_metric_inputs(
                    systems,
                    predictions,
                    targets,
                    extra_data,
                    atomic_basis_reverse_transform,
                    model,
                )
            )
            self._update_metric_calculators(
                rmse_calculator,
                mae_calculator,
                scaled_predictions,
                scaled_targets,
                metric_extra_data,
                "validation",
            )

        return {
            "loss": val_loss,
            **self._finalize_metric_info(
                rmse_calculator,
                mae_calculator,
                per_structure_targets,
                device,
            ),
        }

    def train(
        self,
        model: EPET,
        dtype: torch.dtype,
        devices: List[torch.device],
        train_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        val_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        checkpoint_dir: str,
    ) -> None:
        coefficient_l2_weight, basis_gram_weight = self._regularization_weights()
        if not self._requires_custom_training_path(model):
            super().train(
                model, dtype, devices, train_datasets, val_datasets, checkpoint_dir
            )
            return

        self._validate_custom_training_path(dtype)

        # This loop mirrors PET's single-process path. Local differences are limited
        # to the optional tensor-basis optimizer group, E-PET regularizers, and the
        # default-off atomic-basis irrep-balanced loss.
        device = devices[0]
        logging.info(f"Training on device {device} with dtype {dtype}")

        model = cast(EPET, _apply_finetuning_if_requested(model, self.hypers))
        _move_model_to_training_device(model, device, dtype)

        train_targets = model.dataset_info.targets
        (
            atomic_basis_reverse_transform,
            train_dataloader,
            val_dataloader,
        ) = self._build_custom_dataloaders(
            model, device, train_datasets, val_datasets
        )
        custom_loss_config, custom_loss_fn, normal_train_targets, loss_fn = (
            self._build_custom_losses(model)
        )
        _log_loss_metadata(loss_fn, custom_loss_fn, custom_loss_config)

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
            finalized_train_info = self._run_training_epoch(
                model,
                train_dataloader,
                train_targets,
                normal_train_targets,
                loss_fn,
                custom_loss_fn,
                atomic_basis_reverse_transform,
                optimizer,
                lr_scheduler,
                coefficient_l2_weight,
                basis_gram_weight,
                per_structure_targets,
                dtype,
                device,
            )
            finalized_val_info = self._run_validation_epoch(
                model,
                val_dataloader,
                train_targets,
                normal_train_targets,
                loss_fn,
                custom_loss_fn,
                atomic_basis_reverse_transform,
                per_structure_targets,
                dtype,
                device,
            )

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

            self._update_best_model(model, optimizer, finalized_val_info, epoch)
            self._save_epoch_checkpoint(
                model, optimizer, lr_scheduler, checkpoint_dir, epoch
            )

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
