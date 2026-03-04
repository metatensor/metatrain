import copy
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union, cast

import numpy as np
import torch
from metatomic.torch import ModelOutput
from metatomic.torch.symmetrized_model import (
    _choose_quadrature,
    _compute_real_wigner_matrices,
    _rotations_from_angles,
    decompose_tensors,
    evaluate_model_over_grid,
    get_euler_angles_quadrature,
    symmetrize_over_grid,
)
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler

from metatrain.utils.abc import ModelInterface, TrainerInterface
from metatrain.utils.additive import get_remove_additive_transform
from metatrain.utils.augmentation import RotationalAugmenter
from metatrain.utils.data import (
    CollateFn,
    CombinedDataLoader,
    Dataset,
    get_num_workers,
    unpack_batch,
    validate_num_workers,
)
from metatrain.utils.distributed.batch_utils import should_skip_batch
from metatrain.utils.distributed.distributed_data_parallel import (
    DistributedDataParallel,
)
from metatrain.utils.distributed.slurm import DistributedEnvironment
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
from metatrain.utils.scaler import get_remove_scale_transform
from metatrain.utils.transfer import batch_to

from . import checkpoints
from .documentation import TrainerHypers
from .model import PET
from .modules.finetuning import apply_finetuning_strategy


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


# Known decomposed target prefixes: maps the decomposed key prefix (before _l*)
# back to the original model output name
_DECOMPOSED_PREFIXES = {
    "energy_l": "energy",
    "forces_l": "forces",
    "non_conservative_forces_l": "non_conservative_forces",
    "stress_l": "stress",
    "non_conservative_stress_l": "non_conservative_stress",
}


def _var_key_to_metric_name(var_key: str, per_atom: bool) -> str:
    """Convert a variance key from ``symmetrize_over_grid`` to a ``MetricLogger``
    metric name.

    The metric name has the format ``"{target} {description}"`` so that
    ``MetricLogger`` can look up the target in ``model_outputs`` for units.

    :param var_key: key from ``symmetrize_over_grid`` ending in ``_var``
    :param per_atom: whether the metric has been divided by the number of atoms

    Examples::

        "energy_l0_var", True   -> "energy L0 std (per atom)"
        "non_conservative_forces_l1_var", False -> "non_conservative_forces L1 std"
        "my_custom_target_var", True  -> "my_custom_target std (per atom)"
    """
    per_atom_suffix = " (per atom)" if per_atom else ""

    # Split off per-block suffix like " (o3_lambda=0,...)" if present
    block_suffix = ""
    if " (" in var_key:
        var_key, block_suffix = var_key.split(" (", 1)
        block_suffix = " (" + block_suffix

    # Try to match a known decomposed prefix
    for prefix, target_name in _DECOMPOSED_PREFIXES.items():
        if var_key.startswith(prefix) and var_key.endswith("_var"):
            middle = var_key[len(prefix) : -len("_var")]
            return f"{target_name} L{middle} std{per_atom_suffix}{block_suffix}"

    # Generic target: "my_target_var" -> "my_target std ..."
    if var_key.endswith("_var"):
        target_name = var_key[: -len("_var")]
        return f"{target_name} std{per_atom_suffix}{block_suffix}"

    return var_key + block_suffix


class Trainer(TrainerInterface[TrainerHypers]):
    __checkpoint_version__ = 12

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

        logging.info("Calculating composition weights")
        model.additive_models[0].train_model(  # this is the composition model
            train_datasets,
            model.additive_models[1:],
            self.hypers["batch_size"],
            is_distributed,
            self.hypers["atomic_baseline"],
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
        extra_data_info = dataset_info.extra_data
        rotational_augmenter = RotationalAugmenter(
            target_info_dict=train_targets, extra_data_info_dict=extra_data_info
        )
        requested_neighbor_lists = get_requested_neighbor_lists(model)
        collate_fn_train = CollateFn(
            target_keys=list(train_targets.keys()),
            callables=[
                rotational_augmenter.apply_random_augmentations,
                get_system_with_neighbor_lists_transform(requested_neighbor_lists),
                get_remove_additive_transform(additive_models, train_targets),
                get_remove_scale_transform(scaler),
            ],
            batch_atom_bounds=self.hypers["batch_atom_bounds"],
        )
        collate_fn_val = CollateFn(
            target_keys=list(train_targets.keys()),
            callables=[  # no augmentation for validation
                get_system_with_neighbor_lists_transform(requested_neighbor_lists),
                get_remove_additive_transform(additive_models, train_targets),
                get_remove_scale_transform(scaler),
            ],
            batch_atom_bounds=self.hypers["batch_atom_bounds"],
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
        loss_hypers = cast(Dict[str, LossSpecification], self.hypers["loss"])  # mypy
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
                # Skip None batches (those outside batch_atom_bounds)
                if should_skip_batch(batch, is_distributed, device):
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

                # average by the number of atoms
                predictions = average_by_num_atoms(
                    predictions, systems, per_structure_targets
                )
                targets = average_by_num_atoms(targets, systems, per_structure_targets)
                train_loss_batch = loss_fn(predictions, targets, extra_data)

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

            with torch.set_grad_enabled(
                any(target_info.gradients for target_info in train_targets.values())
            ):  # keep gradients on if any of the targets require them
                val_loss = 0.0
                for batch in val_dataloader:
                    # Skip None batches (those outside batch_atom_bounds)
                    if should_skip_batch(batch, is_distributed, device):
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

                    # average by the number of atoms
                    predictions = average_by_num_atoms(
                        predictions, systems, per_structure_targets
                    )
                    targets = average_by_num_atoms(
                        targets, systems, per_structure_targets
                    )
                    val_loss_batch = loss_fn(predictions, targets, extra_data)

                    if is_distributed:
                        # sum the loss over all processes
                        torch.distributed.all_reduce(val_loss_batch)
                    val_loss += val_loss_batch.item()

                    scaled_predictions = (
                        model.module if is_distributed else model
                    ).scaler(systems, predictions)
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

        # Unsupervised equivariance training (after supervised training)
        if self.hypers["unsupervised_epochs"] > 0:
            logging.info(
                f"Starting {self.hypers['unsupervised_epochs']} unsupervised "
                "epochs to minimize the equivariance error"
            )
            self._train_unsupervised(
                model=(model.module if is_distributed else model),
                train_dataloader=train_dataloader,
                device=device,
                dtype=dtype,
                is_distributed=is_distributed,
                checkpoint_dir=checkpoint_dir,
                rank=rank,
            )

        if is_distributed:
            torch.distributed.destroy_process_group()

    def _train_unsupervised(
        self,
        model: PET,
        train_dataloader: "CombinedDataLoader",
        device: torch.device,
        dtype: torch.dtype,
        is_distributed: bool,
        checkpoint_dir: str,
        rank: int = 0,
    ) -> None:
        """Run unsupervised equivariance training after supervised training.

        Evaluates the model on an O(3) quadrature grid for each training
        structure and minimizes the equivariance variance (which is zero for
        a perfectly equivariant model).
        """
        l_max = self.hypers["unsupervised_l_max"]
        n_unsupervised_epochs = self.hypers["unsupervised_epochs"]
        lr = self.hypers["unsupervised_learning_rate"]
        if lr is None:
            lr = self.hypers["learning_rate"]

        # Build the O(3) quadrature grid
        lebedev_order, n_inplane = _choose_quadrature(l_max)
        alpha, beta, gamma, w_so3 = get_euler_angles_quadrature(
            lebedev_order, n_inplane
        )
        so3_weights = torch.from_numpy(w_so3).to(device=device, dtype=dtype)
        so3_rotations = torch.from_numpy(
            _rotations_from_angles(alpha, beta, gamma).as_matrix()
        ).to(device=device, dtype=dtype)
        angles_inv = (np.pi - gamma, beta, np.pi - alpha)
        so3_inverse_rotations = torch.from_numpy(
            _rotations_from_angles(*angles_inv).as_matrix()
        ).to(device=device, dtype=dtype)

        # Determine max_o3_lambda_target from the model's targets
        train_targets = model.dataset_info.targets
        max_o3_lambda_target = 0
        for target_info in train_targets.values():
            if target_info.is_cartesian:
                tensor_rank = len(target_info.layout.block(0).components)
                max_o3_lambda_target = max(max_o3_lambda_target, tensor_rank)
            elif target_info.is_spherical:
                ell = (len(target_info.layout.block(0).components[0]) - 1) // 2
                max_o3_lambda_target = max(max_o3_lambda_target, ell)

        # Compute Wigner D matrices for back-rotation
        raw_wigner = _compute_real_wigner_matrices(max_o3_lambda_target, angles_inv)
        wigner_D_inverse: Dict[int, torch.Tensor] = {}
        for ell, D in raw_wigner.items():
            if isinstance(D, np.ndarray):
                D = torch.from_numpy(D)
            wigner_D_inverse[ell] = D.to(device=device, dtype=dtype)

        # Build the outputs dict from train_targets (skip gradient-derived targets)
        outputs: Dict[str, ModelOutput] = {}
        for name, info in train_targets.items():
            if name.endswith("_gradients"):
                continue
            outputs[name] = ModelOutput(
                quantity=info.quantity,
                unit=info.unit,
                per_atom=info.per_atom,
                description=info.description,
            )

        # Fresh optimizer
        if self.hypers["weight_decay"] is not None:
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=self.hypers["weight_decay"]
            )
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Fresh LR scheduler (cosine with warmup)
        steps_per_epoch = len(train_dataloader)
        fake_hypers = copy.deepcopy(self.hypers)
        fake_hypers["num_epochs"] = n_unsupervised_epochs
        lr_scheduler = get_scheduler(
            optimizer,
            fake_hypers,
            steps_per_epoch,
        )

        n_rotations = len(so3_rotations)
        logging.info(
            f"Unsupervised equivariance training: {n_unsupervised_epochs} epochs, "
            f"L_max={l_max}, {n_rotations} SO(3) rotations x 2 inversions = "
            f"{2 * n_rotations} evaluations per structure"
        )

        metric_logger: Optional[MetricLogger] = None
        per_atom_keys: set = set()  # keys where "atom" is NOT in samples

        for epoch in range(n_unsupervised_epochs):
            epoch_loss = 0.0
            # Accumulate [sum_of_per_atom_variance, count] per key
            std_accumulators: Dict[str, List[float]] = {}
            n_batches = 0

            for batch in train_dataloader:
                if should_skip_batch(batch, is_distributed, device):
                    continue

                optimizer.zero_grad()

                systems, _, _ = unpack_batch(batch)
                systems, _, _ = batch_to(systems, {}, {}, dtype=dtype, device=device)
                num_atoms = torch.tensor(
                    [len(s) for s in systems], device=device, dtype=dtype
                )

                with torch.enable_grad():
                    backtransformed = evaluate_model_over_grid(
                        model,
                        n_rotations,
                        so3_rotations,
                        so3_inverse_rotations,
                        wigner_D_inverse,
                        False,
                        systems,
                        outputs,
                    )
                    # Decompose on the actual data device (GPU by default to preserve
                    # gradients through decompose/symmetrize operations)
                    decompose_device = next(
                        iter(backtransformed.values())
                    ).block(0).values.device
                    decomposed = decompose_tensors(backtransformed, decompose_device)
                    mean_var = symmetrize_over_grid(decomposed, so3_weights)

                # Sum all variance terms as the loss, tracking per-key contributions
                loss = torch.zeros(1, device=device, dtype=dtype)
                for key, tensor in mean_var.items():
                    if not key.endswith("_var"):
                        continue
                    key_sum = torch.zeros(1, device=device, dtype=dtype)
                    for block_i, block in enumerate(tensor.blocks()):
                        block_sum = block.values.sum()
                        key_sum = key_sum + block_sum

                        # For non-decomposed targets with multiple blocks,
                        # log per-block
                        if len(tensor.blocks()) > 1:
                            block_key_str = key
                            bk = tensor.keys[block_i]
                            block_key_str += " ("
                            for bk_name, bk_val in zip(
                                bk.names, bk.values, strict=True
                            ):
                                block_key_str += f"{bk_name}={int(bk_val)},"
                            block_key_str = block_key_str[:-1] + ")"
                            self._accumulate_per_atom_std(
                                std_accumulators,
                                per_atom_keys,
                                block_key_str,
                                block,
                                num_atoms,
                            )

                    loss = loss + key_sum
                    # Accumulate per-atom std for this key (across all blocks)
                    for block in tensor.blocks():
                        self._accumulate_per_atom_std(
                            std_accumulators, per_atom_keys, key, block, num_atoms
                        )

                if is_distributed:
                    for param in model.parameters():
                        loss = loss + 0.0 * param.sum()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.hypers["grad_clip_norm"]
                )
                optimizer.step()
                lr_scheduler.step()

                if is_distributed:
                    torch.distributed.all_reduce(loss)
                epoch_loss += loss.item()
                n_batches += 1

            # Build the metrics dict with readable names
            divisor = max(n_batches, 1)
            metrics: Dict[str, float] = {"loss": epoch_loss / divisor}
            for var_key, (var_sum, count) in std_accumulators.items():
                metric_name = _var_key_to_metric_name(
                    var_key, per_atom=var_key in per_atom_keys
                )
                metrics[metric_name] = math.sqrt(max(var_sum / max(count, 1), 0.0))

            if epoch == 0:
                # MetricLogger uses log10 of initial values to set display
                # width; clamp zeros to avoid log10(0) crash in _get_digits
                initial_metrics = {k: max(v, 1e-20) for k, v in metrics.items()}
                metric_logger = MetricLogger(
                    log_obj=ROOT_LOGGER,
                    dataset_info=model.dataset_info,
                    initial_metrics=[initial_metrics],
                    names=["unsupervised"],
                )
            if epoch % self.hypers["log_interval"] == 0:
                assert metric_logger is not None
                metric_logger.log(
                    metrics=[metrics],
                    epoch=epoch,
                    rank=rank,
                    learning_rate=optimizer.param_groups[0]["lr"],
                )

    @staticmethod
    def _accumulate_per_atom_std(
        accumulators: Dict[str, List[float]],
        per_atom_keys: set,
        key: str,
        block: "TensorBlock",
        num_atoms: torch.Tensor,
    ) -> None:
        """Accumulate variance divided by n_atoms for per-structure blocks.

        For blocks whose samples contain "atom" (e.g. forces), the values are
        already per-atom so they are accumulated as-is. For per-structure blocks
        (e.g. energy, stress), each system's variance is divided by n_atoms^2
        before accumulating, matching the supervised "(per atom)" convention.
        The ``per_atom_keys`` set is updated with keys that were normalized.
        """
        if key not in accumulators:
            accumulators[key] = [0.0, 0]

        if "atom" in block.samples.names:
            accumulators[key][0] += block.values.sum().item()
            accumulators[key][1] += block.values.shape[0]  # n_atoms
        else:
            # Divide each system's variance by n_atoms^2, then sum
            # block.values shape: (n_systems, n_properties)
            n_atoms_sq = (num_atoms**2).view(-1, *[1] * (block.values.ndim - 1))
            per_atom_var = block.values.detach() / n_atoms_sq
            accumulators[key][0] += per_atom_var.sum().item()
            accumulators[key][1] += block.values.shape[0]  # n_systems
            per_atom_keys.add(key)

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
