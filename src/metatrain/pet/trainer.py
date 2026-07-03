import copy
import logging
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Union, cast

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler

from metatomic.torch import System

from metatrain.utils.abc import ModelInterface, TrainerInterface
from metatrain.utils.additive import get_remove_additive_transform
from metatrain.utils.augmentation import RotationalAugmenter
from metatrain.utils.data import (
    CollateFn,
    CombinedDataLoader,
    Dataset,
    MaxAtomDistributedBatchSampler,
    get_num_workers,
    unpack_batch,
    validate_num_workers,
)
from metatrain.utils.data.atomic_basis_helpers import (
    get_prepare_atomic_basis_targets_transform,
)
from metatrain.utils.distributed.batch_utils import should_skip_batch
from metatrain.utils.distributed.distributed_data_parallel import (
    DistributedDataParallel,
)
from metatrain.utils.distributed.slurm import initialize_slurm_nccl_process_group
from metatrain.utils.evaluate_model import evaluate_model
from metatrain.utils.io import check_file_extension
from metatrain.utils.logging import ROOT_LOGGER, MetricLogger
from metatrain.utils.loss import LossAggregator, LossSpecification
from metatrain.utils.metrics import MAEAccumulator, RMSEAccumulator, get_selected_metric
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists_transform,
)
from metatrain.utils.per_atom import average_by_num_atoms
from metatensor.torch import TensorBlock, TensorMap
from metatrain.utils.loss import DensityMSELossViaC
from metatrain.utils.pyscf_loss import (
    RaggedMetricMatrices,
    get_density_fit_constant_transform,
    get_metric_matrices_transform,
    get_overlap_matrices_transform,
    resolve_ri_aux_basis,
    ri_projections_name,
)
from metatrain.utils.scaler import get_remove_scale_transform
from metatrain.utils.system_data import get_system_data_transform
from metatrain.utils.transfer import batch_to

from . import checkpoints
from .documentation import TrainerHypers
from .model import PET
from .modules.finetuning import apply_finetuning_strategy


def _unpack_batch_to(batch, dtype, device):
    """Unpack a batch and move it to ``dtype``/``device``.

    Ragged metric matrices (:py:class:`RaggedMetricMatrices`) are popped out before
    :func:`batch_to`, which is TorchScript-typed ``Dict[str, TensorMap]`` and cannot
    accept them, then moved and reattached. No-op for batches without such entries.
    """
    systems, targets, extra_data = unpack_batch(batch)
    ragged = {
        key: extra_data.pop(key)
        for key in list(extra_data.keys())
        if isinstance(extra_data[key], RaggedMetricMatrices)
    }
    systems, targets, extra_data = batch_to(
        systems, targets, extra_data, dtype=dtype, device=device
    )
    for key, value in ragged.items():
        extra_data[key] = value.to(dtype=dtype, device=device)
    return systems, targets, extra_data


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    train_hypers: TrainerHypers,
    steps_per_epoch: int,
) -> LambdaLR:
    """
    Get a CosineAnnealing learning-rate scheduler with warmup

    :param optimizer: The optimizer for which to create the scheduler.
    :param train_hypers: The training hyperparameters.
    :param steps_per_epoch: The number of steps per epoch.
    :return: The learning rate scheduler.
    """
    total_steps = train_hypers["num_epochs"] * steps_per_epoch
    warmup_steps = int(train_hypers["warmup_fraction"] * total_steps)
    min_lr_ratio = 0.0  # hardcoded for now, could be made configurable in the future

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine decay
            progress = (current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    return scheduler


def _clone_state_dict_to_cpu(state: Any) -> Any:
    """Recursively clone a (possibly nested) state dict, moving tensors to CPU.

    Keeps the "best" checkpoint off the training device instead of holding a second
    copy of every model/optimizer tensor on GPU (as ``copy.deepcopy`` would).
    """
    if isinstance(state, torch.Tensor):
        return state.detach().cpu().clone()
    if isinstance(state, dict):
        return {k: _clone_state_dict_to_cpu(v) for k, v in state.items()}
    if isinstance(state, list):
        return [_clone_state_dict_to_cpu(v) for v in state]
    return copy.deepcopy(state)


class Trainer(TrainerInterface[TrainerHypers]):
    __checkpoint_version__ = 13

    def __init__(self, hypers: TrainerHypers) -> None:
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
        model: PET,
        dtype: torch.dtype,
        devices: List[torch.device],
        train_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        val_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        checkpoint_dir: str,
    ) -> None:
        assert dtype in PET.__supported_dtypes__

        is_distributed = self.hypers["distributed"]
        is_finetune = self.hypers["finetune"]["read_from"] is not None

        if is_distributed:
            if len(devices) > 1:
                raise ValueError(
                    "Requested distributed training with the `multi-gpu` device. "
                    " If you want to run distributed training with PET, please "
                    "set `device` to cuda."
                )
            # the calculation of the device number works both when GPUs on different
            # processes are not visible to each other and when they are
            device, world_size, rank = initialize_slurm_nccl_process_group(
                self.hypers["distributed_port"]
            )
        else:
            rank = 0
            world_size = 1
            device = devices[0]
            # only one device, as we don't support non-distributed multi-gpu for now

        if is_distributed:
            logging.info(f"Training on {world_size} devices with dtype {dtype}")
        else:
            logging.info(f"Training on device {device} with dtype {dtype}")

        # Apply fine-tuning strategy if provided
        if is_finetune:
            assert self.hypers["finetune"]["read_from"] is not None  # for mypy
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
            inherit_heads = self.hypers["finetune"]["inherit_heads"]
            if inherit_heads:
                logging.info(
                    "Inheriting initial weights for heads and last layers for targets: "
                    f"from {list(inherit_heads.values())} to "
                    f"{list(inherit_heads.keys())}"
                )

        # Move the model to the device and dtype:
        model.to(device=device, dtype=dtype)
        # The additive models of PET are always in float64 (to avoid numerical errors in
        # the composition weights, which can be very large).
        for additive_model in model.additive_models:
            additive_model.to(dtype=torch.float64)
        model.scaler.to(dtype=torch.float64)

        # Set up transformations
        dataset_info = model.dataset_info
        train_targets = dataset_info.targets
        extra_data_info = dataset_info.extra_data
        rotational_augmenter = RotationalAugmenter(
            target_info_dict=train_targets, extra_data_info_dict=extra_data_info
        )
        requested_neighbor_lists = get_requested_neighbor_lists(model)
        max_atoms = self.hypers["max_atoms_per_batch"]
        # When max_atoms_per_batch is set, batches are pre-filtered by atom count at
        # construction time, so batch_atom_bounds filtering in the collate function is
        # not needed (and its documented behaviour is to be ignored in this mode).
        batch_atom_bounds = (
            None if max_atoms is not None else self.hypers["batch_atom_bounds"]
        )
        atomic_basis_transform, atomic_basis_reverse_transform = (
            get_prepare_atomic_basis_targets_transform(train_targets, extra_data_info)
        )
        loss_hypers = cast(Dict[str, LossSpecification], self.hypers["loss"])

        logging.info("Calculating composition weights")
        model.additive_models[0].train_model(  # this is the composition model
            train_datasets,
            model.additive_models[1:],
            self.hypers["batch_size"],
            is_distributed,
            self.hypers["atomic_baseline"],
            initial_transforms=[atomic_basis_transform],
        )

        if self.hypers["scale_targets"]:
            logging.info("Calculating scaling weights")
            model.scaler.train_model(
                train_datasets,
                model.additive_models,
                self.hypers["batch_size"],
                is_distributed,
                self.hypers["fixed_scaling_weights"],
                initial_transforms=[atomic_basis_transform],
                per_structure_targets=self.hypers["per_structure_targets"],
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

        # Classify which targets use each RI loss type (needed for in-loop scaling).
        density_overlap_targets: set[str] = {
            name
            for name, spec in loss_hypers.items()
            if spec.get("type") == "density_mse_via_c"
        }
        density_fit_targets: set[str] = {
            name
            for name, spec in loss_hypers.items()
            if spec.get("type") == "density_mse_via_w"
        }
        log_density_loss: bool = bool(self.hypers.get("log_density_loss", False))

        ri_train_transforms, ri_val_transforms = self._get_ri_transforms(
            loss_hypers, dtype
        )

        # Classify which targets use each RI loss type (needed for in-loop scaling).
        density_overlap_targets: set[str] = {
            name
            for name, spec in loss_hypers.items()
            if spec.get("type") == "density_mse_via_c"
        }
        density_fit_targets: set[str] = {
            name
            for name, spec in loss_hypers.items()
            if spec.get("type") == "density_mse_via_w"
        }
        log_density_loss: bool = bool(self.hypers.get("log_density_loss", False))

        ri_train_transforms, ri_val_transforms = self._get_ri_transforms(
            loss_hypers, dtype
        )

        # Create collate functions

        conditioning_keys = list(model.requested_inputs().keys())
        if conditioning_keys:
            splits = [("training", train_datasets), ("validation", val_datasets)]
            for split, datasets in splits:
                if len(datasets[0]) == 0:
                    continue

                fields = datasets[0][0]._asdict()
                missing_keys = [key for key in conditioning_keys if key not in fields]
                if missing_keys:
                    logging.warning(
                        f"System conditioning is enabled but {missing_keys} are not in "
                        f"the {split} data and will fall back to defaults."
                    )

        conditioning_callables = (
            [get_system_data_transform(conditioning_keys)] if conditioning_keys else []
        )

        target_keys = list(train_targets.keys())
        # Shared callables that run after `atomic_basis_transform` (and after
        # rotational augmentation in training).
        base_callables: List[Callable[..., Any]] = [
            get_system_with_neighbor_lists_transform(requested_neighbor_lists),
            *conditioning_callables,
            get_remove_additive_transform(additive_models, train_targets),
            get_remove_scale_transform(scaler),
        ]
        collate_fn_train = CollateFn(
            target_keys=target_keys,
            callables=[
                atomic_basis_transform,
                rotational_augmenter.apply_random_augmentations,
                *ri_train_transforms,
                *base_callables,
            ],
            batch_atom_bounds=batch_atom_bounds,
        )
        collate_fn_val = CollateFn(
            target_keys=target_keys,
            callables=[  # no augmentation for validation
                atomic_basis_transform,
                *ri_val_transforms,
                *base_callables,
            ],
            batch_atom_bounds=batch_atom_bounds,
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

        # On CUDA (especially GH200 unified memory), forking after CUDA init causes
        # workers to inherit GPU memory mappings, inflating per-worker RSS and triggering
        # OOM. Use 'spawn' to start workers as fresh processes instead.
        mp_context = "spawn" if num_workers > 0 and device.type == "cuda" else None

        if mp_context == "spawn":
            # Some container setups (e.g. CSCS daint ml4es) restrict /dev/shm so that
            # ftruncate() fails with EINVAL.  PyTorch's default 'file_descriptor' sharing
            # strategy creates POSIX shared-memory files in /dev/shm; switching to
            # 'file_system' uses regular files in /tmp instead, which always works.
            import torch.multiprocessing as _torch_mp

            _torch_mp.set_sharing_strategy("file_system")

        # Samplers that need set_epoch() called each epoch (may be DistributedSampler
        # or MaxAtomDistributedBatchSampler depending on which path is taken below).
        epoch_samplers: List[
            Union[DistributedSampler, MaxAtomDistributedBatchSampler]
        ] = []

        train_dataloaders = []
        for train_dataset, train_sampler in zip(
            train_datasets, train_samplers, strict=True
        ):
            if max_atoms is not None:
                batch_sampler = MaxAtomDistributedBatchSampler(
                    dataset=train_dataset,
                    max_atoms=max_atoms,
                    min_atoms=self.hypers["min_atoms_per_batch"],
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=True,
                    drop_last=True,
                )
                epoch_samplers.append(batch_sampler)
                train_dataloaders.append(
                    DataLoader(
                        dataset=train_dataset,
                        batch_sampler=batch_sampler,
                        collate_fn=collate_fn_train,
                        num_workers=num_workers,
                        multiprocessing_context=mp_context,
                        persistent_workers=(num_workers > 0),
                    )
                )
            else:
                if len(train_dataset) < self.hypers["batch_size"]:
                    raise ValueError(
                        f"A training dataset has fewer samples "
                        f"({len(train_dataset)}) than the batch size "
                        f"({self.hypers['batch_size']}). "
                        "Please reduce the batch size."
                    )
                if train_sampler is not None:
                    epoch_samplers.append(train_sampler)
                train_dataloaders.append(
                    DataLoader(
                        dataset=train_dataset,
                        batch_size=self.hypers["batch_size"],
                        sampler=train_sampler,
                        shuffle=(train_sampler is None),
                        drop_last=(train_sampler is None),
                        collate_fn=collate_fn_train,
                        num_workers=num_workers,
                        multiprocessing_context=mp_context,
                        persistent_workers=(num_workers > 0),
                    )
                )
        train_dataloader = CombinedDataLoader(train_dataloaders, shuffle=True)

        # Create dataloader for the validation datasets:
        val_dataloaders = []
        for val_dataset, val_sampler in zip(val_datasets, val_samplers, strict=True):
            if max_atoms is not None:
                val_batch_sampler = MaxAtomDistributedBatchSampler(
                    dataset=val_dataset,
                    max_atoms=max_atoms,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=False,
                )
                val_dataloaders.append(
                    DataLoader(
                        dataset=val_dataset,
                        batch_sampler=val_batch_sampler,
                        collate_fn=collate_fn_val,
                        num_workers=num_workers,
                        multiprocessing_context=mp_context,
                        persistent_workers=(num_workers > 0),
                    )
                )
            else:
                val_dataloaders.append(
                    DataLoader(
                        dataset=val_dataset,
                        batch_size=self.hypers["batch_size"],
                        sampler=val_sampler,
                        shuffle=False,
                        drop_last=False,
                        collate_fn=collate_fn_val,
                        num_workers=num_workers,
                        multiprocessing_context=mp_context,
                        persistent_workers=(num_workers > 0),
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

        if self.hypers["weight_decay"] is not None:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.hypers["learning_rate"],
                weight_decay=self.hypers["weight_decay"],
            )
        else:
            optimizer = torch.optim.Adam(
                model.parameters(), lr=self.hypers["learning_rate"]
            )

        if self.optimizer_state_dict is not None:
            # try to load the optimizer state dict, but this is only possible
            # if there are no new targets in the model (new parameters)
            if not (model.module if is_distributed else model).has_new_targets:
                optimizer.load_state_dict(self.optimizer_state_dict)

        # Create a learning rate scheduler
        lr_scheduler = get_scheduler(optimizer, self.hypers, len(train_dataloader))

        if self.scheduler_state_dict is not None:
            # same as the optimizer, try to load the scheduler state dict
            if not (model.module if is_distributed else model).has_new_targets:
                lr_scheduler.load_state_dict(self.scheduler_state_dict)

        per_structure_targets = self.hypers["per_structure_targets"]

        # Create the validation density-loss evaluator (Δc^T S Δc) if needed.
        # One instance per RI target; it is only evaluated on the validation set.
        if log_density_loss:
            if not any(loss_hypers):
                raise ValueError(
                    "'log_density_loss' requires at least one RI target in the loss."
                )
            # Use the first (and typically only) RI target for the density metric.
            _ri_target = next(iter(loss_hypers))
            density_loss_fn = DensityMSELossViaC(
                name=_ri_target,
                gradient=None,
                weight=1.0,
                reduction="mean",
                metric="overlap",
            )

        # Log the initial learning rate:
        logging.info(f"Base learning rate: {self.hypers['learning_rate']}")

        start_epoch = 0 if self.epoch is None else self.epoch + 1

        # Train the model:
        if self.best_metric is None:
            self.best_metric = float("inf")
        logging.info("Starting training")
        epoch = start_epoch

        for epoch in range(start_epoch, self.hypers["num_epochs"]):
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
                # Skip None batches (those outside batch_atom_bounds)
                if should_skip_batch(batch, is_distributed, device):
                    continue

                optimizer.zero_grad()

                systems, targets, extra_data = _unpack_batch_to(batch, dtype, device)
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

                # Rescale predictions/targets to the physical units expected by the
                # active RI loss type (density_overlap or density_fit).  For direct-c
                # this is a no-op.
                loss_predictions, loss_targets = self._prepare_for_ri_loss(
                    predictions,
                    targets,
                    systems,
                    model.module if is_distributed else model,
                    density_overlap_targets,
                    density_fit_targets,
                    train_targets,
                )
                train_loss_batch = loss_fn(loss_predictions, loss_targets, extra_data)

                if is_distributed:
                    # make sure all parameters contribute to the gradient calculation
                    # to make torch DDP happy
                    for param in model.parameters():
                        train_loss_batch += 0.0 * param.sum()

                train_loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.hypers["grad_clip_norm"]
                )
                optimizer.step()
                lr_scheduler.step()

                if is_distributed:
                    # sum the loss over all processes
                    torch.distributed.all_reduce(train_loss_batch)
                train_loss += train_loss_batch.item()

                # Reapply scales and accumulate quantities for computing train metrics,
                # but only if this is an epoch to log
                if epoch == start_epoch or epoch % self.hypers["log_interval"] == 0:
                    scaled_predictions = (
                        model.module if is_distributed else model
                    ).scaler(
                        systems,
                        predictions,
                        remove=False,
                        use_per_target_scales=True,
                        use_per_property_scales=False,
                    )
                    scaled_targets = (model.module if is_distributed else model).scaler(
                        systems,
                        targets,
                        remove=False,
                        use_per_target_scales=True,
                        use_per_property_scales=False,
                    )

                    if self.hypers["log_separate_blocks"]:
                        # if any atomic basis outputs are present and metrics are to be
                        # reported per-block, reverse the transform (i.e. sparsify)
                        # before calculating metrics
                        systems, scaled_targets, extra_data = (
                            atomic_basis_reverse_transform(
                                systems, scaled_targets, extra_data
                            )
                        )
                        systems, scaled_predictions, _ = atomic_basis_reverse_transform(
                            systems, scaled_predictions, {}
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

            with torch.set_grad_enabled(
                any(target_info.gradients for target_info in train_targets.values())
            ):  # keep gradients on if any of the targets require them
                val_loss = 0.0
                val_density_loss = 0.0
                for batch in val_dataloader:
                    # Skip None batches (those outside batch_atom_bounds)
                    if should_skip_batch(batch, is_distributed, device):
                        continue

                    systems, targets, extra_data = _unpack_batch_to(
                        batch, dtype, device
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
                    targets = average_by_num_atoms(
                        targets, systems, per_structure_targets
                    )

                    # Apply per-property scales to the predictions before loss
                    # computation. The targets from the dataloader have only been scaled
                    # per-target, and not per-property. This transformation only applies
                    # to targets with per-property scales (i.e. multiple blocks or
                    # multiple properties), and leaves the others unchanged.
                    predictions = (model.module if is_distributed else model).scaler(
                        systems,
                        predictions,
                        remove=False,
                        use_per_target_scales=False,
                        use_per_property_scales=True,
                    )

                    loss_predictions, loss_targets = self._prepare_for_ri_loss(
                        predictions,
                        targets,
                        systems,
                        model.module if is_distributed else model,
                        density_overlap_targets,
                        density_fit_targets,
                        train_targets,
                    )
                    val_loss_batch = loss_fn(loss_predictions, loss_targets, extra_data)

                    if is_distributed:
                        # sum the loss over all processes
                        torch.distributed.all_reduce(val_loss_batch)
                    val_loss += val_loss_batch.item()

                    # Reapply scales and accumulate quantities for computing val
                    # metrics. This is done for every epoch as validation metrics are
                    # needed for model selection.
                    # scaled_predictions = (c_ML - CM),  scaled_targets = (c_RI - CM).
                    scaled_predictions = (
                        model.module if is_distributed else model
                    ).scaler(
                        systems,
                        predictions,
                        remove=False,
                        use_per_target_scales=True,
                        use_per_property_scales=False,
                    )
                    scaled_targets = (model.module if is_distributed else model).scaler(
                        systems,
                        targets,
                        remove=False,
                        use_per_target_scales=True,
                        use_per_property_scales=False,
                    )

                    if self.hypers["log_separate_blocks"]:
                        # if any atomic basis outputs are present and metrics are to be
                        # reported per-block, reverse the transform (i.e. sparsify)
                        # before calculating metrics
                        systems, scaled_targets, extra_data = (
                            atomic_basis_reverse_transform(
                                systems, scaled_targets, extra_data
                            )
                        )
                        systems, scaled_predictions, _ = atomic_basis_reverse_transform(
                            systems, scaled_predictions, {}
                        )

                    val_rmse_calculator.update(
                        scaled_predictions, scaled_targets, extra_data
                    )
                    if self.hypers["log_mae"]:
                        val_mae_calculator.update(
                            scaled_predictions, scaled_targets, extra_data
                        )

                    if log_density_loss:
                        # Compute the real-space density L2 metric:
                        # L = Δc^T S Δc, where Δc = c_ML - c_RI (CM-removed).
                        # scaled_predictions = c_ML - CM, scaled_targets = c_RI - CM,
                        # so their difference is Δc.  Overlap S is in extra_data when
                        # ri_aux_basis is set and log_density_loss is True.
                        density_batch = density_loss_fn(
                            scaled_predictions, scaled_targets, extra_data
                        )
                        if is_distributed:
                            torch.distributed.all_reduce(density_batch)
                        val_density_loss += density_batch.item()

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
                finalized_train_info = {
                    "loss": train_loss,
                    **finalized_train_info,
                }
            finalized_val_info = {
                "loss": val_loss,
                **finalized_val_info,
            }
            if log_density_loss:
                finalized_val_info["density_loss"] = val_density_loss

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
                self.best_model_state_dict = _clone_state_dict_to_cpu(
                    (model.module if is_distributed else model).state_dict()
                )
                self.best_epoch = epoch
                self.best_optimizer_state_dict = _clone_state_dict_to_cpu(
                    optimizer.state_dict()
                )

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

    def _get_ri_transforms(
        self,
        loss_hypers: Dict[str, LossSpecification],
        dtype: torch.dtype = torch.float64,
    ) -> tuple[list[Callable], list[Callable]]:
        """
        Build the collate transforms required for RI density losses.

        Returns ``(train_transforms, val_transforms)``.  The validation transforms
        always include the overlap-matrix computation when ``log_density_loss`` is
        enabled (so the density metric is available regardless of training loss type).

        Transforms run *before* CM removal and scaling, which is necessary for the
        density-fit constant pre-computation.

        :param dtype: dtype of the metric matrices. The matrices are stored ragged
            (:py:class:`RaggedMetricMatrices`) and carried raw past ``save_buffer``, so —
            unlike the old packed-TensorMap path — they can be the model dtype
            (e.g. float32), halving their memory/transport. Casting matches the recast
            ``batch_to`` applied before, so training numerics are unchanged.
        """
        ri_loss_types = {"density_mse_via_c", "density_mse_via_w"}
        ri_aux_basis = self.hypers["ri_aux_basis"]
        log_density_loss = bool(self.hypers.get("log_density_loss", False))

        # Check that ri_aux_basis is provided when needed.
        if any(spec.get("type") in ri_loss_types for spec in loss_hypers.values()):
            if ri_aux_basis is None:
                raise ValueError(
                    "PET training with RI density losses ('density_mse_via_c', "
                    "'density_mse_via_w') requires 'ri_aux_basis' to be set."
                )

        if ri_aux_basis is None and not log_density_loss:
            return [], []

        if ri_aux_basis is None:
            # log_density_loss=True but no loss hypers use RI — nothing to compute.
            return [], []

        # Build per-metric, per-target → aux_basis mappings.
        # The overlap metric is always needed for the log_density_loss validation metric.
        metric_targets: dict[str, dict[str, str]] = {}  # metric → {target: aux_basis}
        # Maps density_mse_via_w target name → the extra_data key for its projections.
        density_fit_target_to_proj_key: dict[str, str] = {}

        for target_name, target_spec in loss_hypers.items():
            loss_type = target_spec.get("type", "mse")
            if loss_type not in ri_loss_types:
                if not log_density_loss:
                    continue
            metric = (
                target_spec.get("metric", "overlap")
                if loss_type in ri_loss_types
                else "overlap"
            )
            aux_basis = resolve_ri_aux_basis(
                target_name,
                (
                    cast(str, ri_aux_basis)
                    if isinstance(ri_aux_basis, str)
                    else cast(dict, ri_aux_basis)
                ),
            )
            metric_targets.setdefault(metric, {})[target_name] = aux_basis
            if loss_type == "density_mse_via_w":
                proj_key = target_spec.get(
                    "projections_key", ri_projections_name(target_name)
                )
                density_fit_target_to_proj_key[target_name] = proj_key

        # Always ensure overlap matrices are available for the density-loss metric.
        if log_density_loss:
            for target_name in loss_hypers:
                aux_basis = resolve_ri_aux_basis(
                    target_name,
                    (
                        cast(str, ri_aux_basis)
                        if isinstance(ri_aux_basis, str)
                        else cast(dict, ri_aux_basis)
                    ),
                )
                metric_targets.setdefault("overlap", {})[target_name] = aux_basis

        if not metric_targets:
            return [], []

        # Build transform lists: one matrix transform per metric, plus constant.
        def _build_transforms(include_constant: bool) -> list[Callable]:
            transforms: list[Callable] = []
            for metric, targets_map in metric_targets.items():
                transforms.append(
                    get_metric_matrices_transform(targets_map, metric, dtype)
                )
            if include_constant and density_fit_target_to_proj_key:
                # Must run before CM removal and scaling.
                transforms.append(
                    get_density_fit_constant_transform(density_fit_target_to_proj_key)
                )
            return transforms

        train_transforms = _build_transforms(include_constant=True)
        val_transforms = _build_transforms(include_constant=True)

        return train_transforms, val_transforms

    def _prepare_for_ri_loss(
        self,
        predictions: Dict[str, Any],
        targets: Dict[str, Any],
        systems: List[System],
        inner_model: torch.nn.Module,
        density_overlap_targets: set[str],
        density_fit_targets: set[str],
        train_targets: Dict[str, Any],
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Rescale predictions (and targets for density_overlap) to the physical units
        required by the active RI loss type before passing to the loss function.

        direct-c:
            No change.  Predictions = (c_ML − CM)/σ_t, targets = (c_RI − CM)/σ_t.

        density_mse_via_c  (L = Δc^T M Δc):
            Multiply both by σ_t.
            Predictions → c_ML − CM,  targets → c_RI − CM.

        density_mse_via_w  (L = c_ML^T M c_ML − 2 c_ML^T w):
            Multiply predictions by σ_t, then add CM back.
            Predictions → c_ML.  Targets (c_RI) are not used by the loss module;
            the reference data are w_RI and the constant, both in extra_data.
        """
        if not density_overlap_targets and not density_fit_targets:
            return predictions, targets

        all_ri_targets = density_overlap_targets | density_fit_targets
        loss_predictions = dict(predictions)
        loss_targets = dict(targets)

        # Apply σ_t to predictions for all RI targets that need physical-unit rescaling.
        for target_name in all_ri_targets:
            if target_name not in predictions:
                continue
            rescaled = inner_model.scaler(
                systems,
                {target_name: predictions[target_name]},
                remove=False,
                use_per_target_scales=True,
                use_per_property_scales=False,
            )
            loss_predictions[target_name] = rescaled[target_name]

        # Apply σ_t to targets for density_overlap (so both sides are CM-removed).
        for target_name in density_overlap_targets:
            if target_name not in targets:
                continue
            rescaled = inner_model.scaler(
                systems,
                {target_name: targets[target_name]},
                remove=False,
                use_per_target_scales=True,
                use_per_property_scales=False,
            )
            loss_targets[target_name] = rescaled[target_name]

        # Add CM back to predictions for density_fit.
        if density_fit_targets:
            loss_predictions = self._add_composition_model_contribution(
                inner_model,
                systems,
                loss_predictions,
                train_targets,
                density_fit_targets,
            )

        return loss_predictions, loss_targets

    def _add_composition_model_contribution(
        self,
        inner_model: torch.nn.Module,
        systems: List[System],
        predictions: Dict[str, Any],
        train_targets: Dict[str, Any],
        target_names: set[str],
    ) -> Dict[str, Any]:
        """
        Add composition-model (CM) contributions back to predictions.

        This reverses the CM-removal applied by the collate pipeline, giving
        fully-reconstructed RI coefficients c_ML = (c_ML − CM) + CM.
        CM values are detached so no gradient flows through them.
        """
        cm_outputs = {
            k: train_targets[k]
            for k in target_names
            if k in train_targets and k in inner_model.additive_models[0].outputs
        }
        if not cm_outputs:
            return predictions

        with torch.no_grad():
            cm_contribution = evaluate_model(
                inner_model.additive_models[0],
                systems,
                cm_outputs,
                is_training=False,
            )

        new_predictions = dict(predictions)
        for target_name, cm_map in cm_contribution.items():
            if target_name not in predictions:
                continue
            pred_map = predictions[target_name]
            new_blocks = []
            for key in pred_map.keys:
                pred_block = pred_map.block(key)
                new_values = pred_block.values
                if key in cm_map.keys:
                    cm_block = cm_map.block(key)
                    cm_values = cm_block.values.detach().to(
                        device=pred_block.values.device,
                        dtype=pred_block.values.dtype,
                    )
                    new_values = pred_block.values + cm_values
                new_blocks.append(
                    TensorBlock(
                        values=new_values,
                        samples=pred_block.samples,
                        components=pred_block.components,
                        properties=pred_block.properties,
                    )
                )
            new_predictions[target_name] = TensorMap(pred_map.keys, new_blocks)

        return new_predictions

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
            trainer.epoch = None  # interpreted as zero in the training loop
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
