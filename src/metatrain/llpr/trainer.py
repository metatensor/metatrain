import copy
import logging
import math
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Optional, Union, cast

import torch
from metatensor.torch import TensorBlock, TensorMap
from metatomic.torch import ModelOutput
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler

from metatrain.utils.abc import (
    ModelInterface,
    TrainerInterface,
    common_upgrade_checkpoint,
)
from metatrain.utils.augmentation import O3Augmenter
from metatrain.utils.data import (
    CollateFn,
    CombinedDataLoader,
    Dataset,
    DatasetInfo,
    build_train_dataloaders,
    build_val_dataloaders,
    get_num_workers,
    unpack_batch,
    validate_num_workers,
)
from metatrain.utils.distributed.distributed_data_parallel import (
    DistributedDataParallel,
)
from metatrain.utils.distributed.slurm import (
    initialize_slurm_nccl_process_group,
    resolve_distributed,
)
from metatrain.utils.evaluate_model import evaluate_model
from metatrain.utils.io import check_file_extension, model_from_checkpoint
from metatrain.utils.logging import ROOT_LOGGER, MetricLogger
from metatrain.utils.loss import LossAggregator, LossSpecification
from metatrain.utils.metrics import MAEAccumulator, RMSEAccumulator, get_selected_metric
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists_transform,
)
from metatrain.utils.per_atom import average_by_num_atoms
from metatrain.utils.transfer import batch_to
from metatrain.utils.wrapper import MetatrainModel

from . import checkpoints
from .calibration import GaussianCRPSCalibrator, RatioCalibrator
from .documentation import ModelHypers, TrainerHypers
from .model import LLPRUncertaintyModel


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
    assert train_hypers["num_epochs"] is not None
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


class Trainer(TrainerInterface[TrainerHypers, ModelHypers]):
    __checkpoint_version__ = 7

    def __init__(self, hypers: TrainerHypers) -> None:
        super().__init__(hypers)

        self.optimizer_state_dict: Optional[Dict[str, Any]] = None
        self.scheduler_state_dict: Optional[Dict[str, Any]] = None
        self.epoch: Optional[int] = None
        self.best_epoch: Optional[int] = None
        self.best_metric: Optional[float] = None
        self.best_model_state_dict: Optional[Dict[str, Any]] = None
        self.best_optimizer_state_dict: Optional[Dict[str, Any]] = None

    def setup(
        self, model_hypers: ModelHypers, dataset_info: DatasetInfo
    ) -> MetatrainModel[LLPRUncertaintyModel]:

        if self.hypers["model_checkpoint"] is None:
            raise ValueError(
                "A model checkpoint must be provided to train the LLPR "
                "(model_checkpoint, under training, in the hypers)"
            )

        llpr_model = LLPRUncertaintyModel(model_hypers, dataset_info)

        wrapped_model_checkpoint_path = self.hypers["model_checkpoint"]
        checkpoint = torch.load(
            wrapped_model_checkpoint_path, weights_only=False, map_location="cpu"
        )
        wrapped_model = model_from_checkpoint(checkpoint, "export")
        llpr_model.set_wrapped_model(wrapped_model)

        return MetatrainModel(
            core=llpr_model,
            additive_models=[],
            scaler=None,
            dataset_info=dataset_info,
        )

    def restart(
        self,
        model: MetatrainModel[LLPRUncertaintyModel],
        dataset_info: DatasetInfo,
        model_hypers: ModelHypers,
    ) -> MetatrainModel[LLPRUncertaintyModel]:
        model.restart(dataset_info, model_hypers=model_hypers)
        return model

    def train(
        self,
        model: MetatrainModel[LLPRUncertaintyModel],
        dtype: torch.dtype,
        devices: List[torch.device],
        train_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        val_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        checkpoint_dir: str,
    ) -> None:
        # we begin by loading start_epoch to determine if restarting or not
        start_epoch = 0 if self.epoch is None else self.epoch + 1

        # Get LLPR model from the MetatrainModel class
        llpr_model = model.core
        # And get the wrapped model from the LLPR model
        wrapped_model = llpr_model.model

        is_distributed = resolve_distributed(self.hypers.get("distributed"))

        # For the initial LLPR calibration, distributed training can be used
        if is_distributed:
            if len(devices) > 1:
                raise ValueError(
                    "Requested distributed training with the `multi-gpu` device. "
                    " If you want to run distributed training with LLPR, please "
                    "set `device` to cuda."
                )
            # Initialize distributed environment for calibration
            device, world_size, rank = initialize_slurm_nccl_process_group(
                self.hypers["distributed_port"]
            )
            logging.info(f"Initialized distributed training on {world_size} devices")
        else:
            device = devices[0]
            rank = 0

        # check device and dtype against wrapped model class
        supported_devices = wrapped_model.core.__class__.__supported_devices__
        if device.type not in supported_devices:
            raise ValueError(
                f"Device {device} not supported. "
                f"Supported devices are {supported_devices}"
            )
        supported_types = wrapped_model.core.__class__.__supported_dtypes__
        if dtype not in supported_types:
            raise ValueError(
                f"dtype {dtype} not supported by the wrapped model. "
                f"Supported dtypes are {supported_types}"
            )

        if is_distributed:
            logging.info(f"Training on {world_size} devices with dtype {dtype}")
        else:
            logging.info(f"Training on device {device} with dtype {dtype}")

        # Move the model to the device and dtype:
        model.to(device=device, dtype=dtype)

        if start_epoch == 0:
            logging.info("Computing LLPR covariance matrix")
            compute_covariance(
                llpr_model, train_datasets, self.hypers["batch_size"], is_distributed
            )
            logging.info("Computing Cholesky decomposition of the covariance matrix")
            llpr_model.compute_cholesky_decomposition(self.hypers["regularizer"])
            logging.info("Calibrating LLPR uncertainties")
            calibrate(
                llpr_model,
                val_datasets,
                self.hypers["batch_size"],
                is_distributed,
                self.hypers["calibration_method"],
            )
            logging.info("Generating LLPR ensemble members")
            llpr_model.generate_ensemble()
            logging.info("LLPR complete!")

        if self.hypers["num_epochs"] is None:
            if is_distributed:
                torch.distributed.destroy_process_group()
            return
        else:
            logging.info("`num_epochs` is set: starting LLPR ensemble weight training")

        # Continue with ensemble training if num_epochs is not None
        # (distributed environment is already initialized if needed)
        world_size = torch.distributed.get_world_size() if is_distributed else 1

        logging.info("Starting gradient-based training for LLPR ensemble calibration")

        # Re-create the dataloaders to make them shuffle and augment the data
        train_targets = model.dataset_info.targets
        extra_data_info = model.dataset_info.extra_data
        rotational_augmenter = O3Augmenter(
            target_info_dict=train_targets, extra_data_info_dict=extra_data_info
        )
        requested_neighbor_lists = get_requested_neighbor_lists(model)
        collate_fn_train = CollateFn(
            target_keys=list(train_targets.keys()),
            callables=[
                rotational_augmenter.apply_random_augmentations,
                get_system_with_neighbor_lists_transform(requested_neighbor_lists),
            ],
        )
        collate_fn_val = CollateFn(
            target_keys=list(train_targets.keys()),
            callables=[  # no augmentation for validation
                get_system_with_neighbor_lists_transform(requested_neighbor_lists),
            ],
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

        max_atoms = self.hypers["max_atoms_per_batch"]

        # Create dataloader for the training datasets:
        train_dataloaders, epoch_samplers = build_train_dataloaders(
            train_datasets=train_datasets,
            train_distributed_samplers=train_samplers,
            collate_fn_train=collate_fn_train,
            batch_size=self.hypers["batch_size"],
            max_atoms_per_batch=max_atoms,
            min_atoms_per_batch=self.hypers["min_atoms_per_batch"],
            num_workers=num_workers,
        )
        train_dataloader = CombinedDataLoader(train_dataloaders, shuffle=True)

        # Create dataloader for the validation datasets:
        val_dataloaders = build_val_dataloaders(
            val_datasets=val_datasets,
            val_distributed_samplers=val_samplers,
            collate_fn_val=collate_fn_val,
            batch_size=self.hypers["batch_size"],
            max_atoms_per_batch=max_atoms,
            num_workers=num_workers,
        )
        val_dataloader = CombinedDataLoader(val_dataloaders, shuffle=False)

        outputs_list = []
        for target_name, target_info in train_targets.items():
            outputs_list.append(target_name)
            for gradient_name in target_info.gradients:
                outputs_list.append(f"{target_name}_{gradient_name}_gradients")

        model = _apply_ensemble_training_strategy(
            model, self.hypers["train_all_parameters"]
        )

        if is_distributed:
            model = DistributedDataParallel(model, device_ids=[device])

        loss_hypers = self.hypers["loss"]
        loss_hypers = cast(Dict[str, LossSpecification], loss_hypers)  # mypy
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

        # Create an optimizer
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
            optimizer.load_state_dict(self.optimizer_state_dict)

        # Create a learning rate scheduler
        lr_scheduler = get_scheduler(optimizer, self.hypers, len(train_dataloader))

        if self.scheduler_state_dict is not None:
            lr_scheduler.load_state_dict(self.scheduler_state_dict)

        per_structure_targets = self.hypers["per_structure_targets"]

        # Log the initial learning rate:
        logging.info(f"Base learning rate: {self.hypers['learning_rate']}")

        # Train the model:
        if self.best_metric is None:
            self.best_metric = float("inf")

        requested_outputs = {}
        model_unwrapped = (model.module if is_distributed else model).core
        for key, value in model_unwrapped.dataset_info.targets.items():
            requested_outputs[key] = model_unwrapped.capabilities.outputs[key]
            requested_outputs[key].sample_kind = value.sample_kind
            ensemble_name = model_unwrapped.targets[key].ensemble_name
            requested_outputs[ensemble_name] = model_unwrapped.capabilities.outputs[
                ensemble_name
            ]
            requested_outputs[ensemble_name].sample_kind = value.sample_kind

        assert self.hypers["num_epochs"] is not None
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
                systems, targets, extra_data = batch_to(
                    systems, targets, extra_data, dtype=dtype
                )

                predictions = evaluate_model(
                    model,
                    systems,
                    requested_outputs,
                    is_training=True,
                )

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

                # Accumulate quantities for computing train metrics,
                # but only if this is an epoch to log
                if epoch == start_epoch or epoch % self.hypers["log_interval"] == 0:
                    predictions = average_by_num_atoms(
                        predictions, systems, per_structure_targets
                    )
                    targets = average_by_num_atoms(
                        targets, systems, per_structure_targets
                    )

                    targets = _drop_gradient_blocks(targets)
                    train_rmse_calculator.update(predictions, targets)
                    if self.hypers["log_mae"]:
                        train_mae_calculator.update(predictions, targets)

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
                for batch in val_dataloader:
                    systems, targets, extra_data = unpack_batch(batch)
                    systems, targets, extra_data = batch_to(
                        systems, targets, extra_data, device=device
                    )
                    systems, targets, extra_data = batch_to(
                        systems, targets, extra_data, dtype=dtype
                    )
                    predictions = evaluate_model(
                        model,
                        systems,
                        requested_outputs,
                        is_training=False,
                    )
                    val_loss_batch = loss_fn(predictions, targets, extra_data)

                    if is_distributed:
                        # sum the loss over all processes
                        torch.distributed.all_reduce(val_loss_batch)
                    val_loss += val_loss_batch.item()

                    # Accumulate quantities for computing val metrics. This is done for
                    # every epoch as validation metrics are needed for model selection
                    predictions = average_by_num_atoms(
                        predictions, systems, per_structure_targets
                    )
                    targets = average_by_num_atoms(
                        targets, systems, per_structure_targets
                    )

                    targets = _drop_gradient_blocks(targets)
                    val_rmse_calculator.update(predictions, targets)
                    if self.hypers["log_mae"]:
                        val_mae_calculator.update(predictions, targets)

                # Compute val metrics:
                finalized_val_info = val_rmse_calculator.finalize(
                    not_per_atom=["positions_gradients"] + per_structure_targets,
                    is_distributed=is_distributed,
                    device=device,
                )
                if self.hypers["log_mae"]:
                    finalized_val_info.update(
                        val_mae_calculator.finalize(
                            not_per_atom=["positions_gradients"]
                            + per_structure_targets,
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
        checkpoint = model.get_checkpoint(self.best_model_state_dict)
        checkpoint.update(
            {
                "trainer_ckpt_version": self.__checkpoint_version__,
                "train_hypers": self.hypers,
                "epoch": self.epoch,
                "optimizer_state_dict": self.optimizer_state_dict,
                "scheduler_state_dict": self.scheduler_state_dict,
                "best_epoch": self.best_epoch,
                "best_metric": self.best_metric,
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
        context: Literal["restart", "finetune"],  # not used at the moment
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
        trainer.best_model_state_dict = MetatrainModel.get_state_dict_from_checkpoint(
            checkpoint, "best_model_state_dict"
        )
        trainer.best_optimizer_state_dict = checkpoint["best_optimizer_state_dict"]
        return trainer

    @classmethod
    def upgrade_checkpoint(
        cls, checkpoint: dict[str, Any], version: Optional[int] = None
    ) -> dict[str, Any]:
        return common_upgrade_checkpoint(
            checkpoint, version, cls.__checkpoint_version__, "trainer", checkpoints
        )


def _drop_gradient_blocks(targets: Dict[str, Any]) -> Dict[str, Any]:
    """Remove gradient blocks from the targets dictionary.

    :param targets: The targets dictionary.
    :return: The targets dictionary without gradient blocks.
    """
    filtered_targets = {}
    for key, value in targets.items():
        new_blocks = []
        for _, b in value.items():
            new_block = TensorBlock(
                values=b.values,
                samples=b.samples,
                components=b.components,
                properties=b.properties,
            )
            new_blocks.append(new_block)
        filtered_targets[key] = TensorMap(value.keys, new_blocks)
    return filtered_targets


def _apply_ensemble_training_strategy(
    model: MetatrainModel[LLPRUncertaintyModel],
    train_all_parameters: bool,
) -> MetatrainModel[LLPRUncertaintyModel]:
    """
    Apply the user-specified ensemble training strategy to the LLPR-wrapped
    model. This function modifies the model in place based on the provided
    trainable parameters.

    :param model: LLPR-wrapped model to be recalibrated.
    :param train_all_parameters: Whether to train all parameters or only the LLPR
        ensemble layers.
    :return: the model with updated trainable parameters.
    """

    # Start by making all parameters trainable
    for param in model.parameters():
        param.requires_grad = True

    if not train_all_parameters:
        # Freeze all parameters of the base model
        for param in model.core.model.parameters():
            param.requires_grad = False

    return model


def compute_covariance(
    model: LLPRUncertaintyModel,
    datasets: List[Union[Dataset, torch.utils.data.Subset]],
    batch_size: int,
    is_distributed: bool,
) -> None:
    """Accumulate the covariance of the last-layer features of each target.

    :param model: the LLPR model.
    :param datasets: the datasets to accumulate the covariance over.
    :param batch_size: the batch size.
    :param is_distributed: whether the datasets are split between processes.
    """
    reference = next(model.model.parameters())
    device, dtype = reference.device, reference.dtype
    sample_kinds = {
        name: model.model.dataset_info.targets[name].sample_kind
        for name in model.targets
    }
    with torch.no_grad():
        for batch in _get_dataloader(model, datasets, batch_size, is_distributed):
            systems, targets, _ = unpack_batch(batch)
            systems = [system.to(device=device, dtype=dtype) for system in systems]
            names = [name for name in targets if name in model.targets]
            outputs = {
                model.targets[name].features_name: ModelOutput(
                    sample_kind=sample_kinds[name]
                )
                for name in names
            }
            features = model(systems, outputs)
            n_atoms = torch.tensor([len(s) for s in systems], device=device)
            for name in names:
                llpr_target = model.targets[name]
                llpr_target.accumulate(
                    features[llpr_target.features_name],
                    n_atoms if sample_kinds[name] == "system" else None,
                )

    if is_distributed:
        torch.distributed.barrier()
        for llpr_target in model.targets.values():
            for covariance in llpr_target.covariances:
                torch.distributed.all_reduce(covariance.covariance)


def calibrate(
    model: LLPRUncertaintyModel,
    datasets: List[Union[Dataset, torch.utils.data.Subset]],
    batch_size: int,
    is_distributed: bool,
    calibration_method: Literal["absolute_residuals", "squared_residuals", "crps"],
) -> None:
    """Fit one calibration factor for each property of each block of each target, so
    that the uncertainties match the errors on ``datasets``.

    :param model: the LLPR model.
    :param datasets: the datasets to calibrate on.
    :param batch_size: the batch size.
    :param is_distributed: whether the datasets are split between processes.
    :param calibration_method: "squared_residuals" and "crps" minimize the negative
        log-likelihood and the continuous ranked probability score of Gaussian
        errors; "absolute_residuals" matches the mean absolute residuals instead,
        which is less sensitive to outliers.
    """
    reference = next(model.model.parameters())
    device, dtype = reference.device, reference.dtype

    # one calibrator for each block of each target, in the same order on all the
    # processes, and holding all the blocks, also those missing from the batches of
    # a process, so that all the processes reduce the same blocks
    blocks = [
        (name, index)
        for name, target in model.targets.items()
        for index in range(len(target.multiplier_offsets) - 1)
    ]
    calibrators: List[Union[RatioCalibrator, GaussianCRPSCalibrator]] = []
    for name, index in blocks:
        calibrator: Union[RatioCalibrator, GaussianCRPSCalibrator]
        if calibration_method == "crps":
            calibrator = GaussianCRPSCalibrator()
        else:
            calibrator = RatioCalibrator(method=calibration_method)
        offsets = model.targets[name].multiplier_offsets
        empty = torch.empty((0, offsets[index + 1] - offsets[index]), device=device)
        calibrator.update(uncertainty_name=name, residuals=empty, uncertainties=empty)
        calibrators.append(calibrator)
    num_samples = torch.zeros(len(blocks), dtype=torch.long, device=device)

    with torch.no_grad():
        for batch in _get_dataloader(model, datasets, batch_size, is_distributed):
            systems, targets, _ = unpack_batch(batch)
            systems = [system.to(device=device, dtype=dtype) for system in systems]
            targets = {
                name: target.to(device=device, dtype=dtype)
                for name, target in targets.items()
                if name in model.targets
            }
            outputs: Dict[str, ModelOutput] = {}
            for name in targets:
                outputs[name] = ModelOutput(
                    sample_kind=model.model.dataset_info.targets[name].sample_kind
                )
                outputs[model.targets[name].uncertainty_name] = outputs[name]
            predictions = model(systems, outputs)

            for name, target in targets.items():
                uncertainty = predictions[model.targets[name].uncertainty_name]
                layout_keys = model.targets[name].layout_keys
                for key, block in predictions[name].items():
                    block_index = layout_keys.position(key)
                    assert block_index is not None
                    i = blocks.index((name, block_index))
                    residuals = block.values - target.block(key).values
                    uncertainties = uncertainty.block(key).values
                    calibrators[i].update(
                        uncertainty_name=name,
                        residuals=residuals.reshape(-1, residuals.shape[-1]),
                        uncertainties=uncertainties.reshape(
                            -1, uncertainties.shape[-1]
                        ),
                    )
                    num_samples[i] += len(block.samples)

    if is_distributed:
        torch.distributed.all_reduce(num_samples)
    for i, (name, block_index) in enumerate(blocks):
        # the blocks missing from the calibration data keep a factor of 1
        if num_samples[i] > 0:
            multipliers = calibrators[i].finalize()[name]
            model.targets[name].set_multipliers(block_index, multipliers)


def _get_dataloader(
    model: LLPRUncertaintyModel,
    datasets: List[Union[Dataset, torch.utils.data.Subset]],
    batch_size: int,
    is_distributed: bool,
) -> CombinedDataLoader:
    """A dataloader over ``datasets`` for a pass of LLPR fitting, without shuffling
    or dropping samples, split between processes if ``is_distributed``.

    :param model: the LLPR model.
    :param datasets: the datasets.
    :param batch_size: the batch size.
    :param is_distributed: whether to split the datasets between processes.
    :return: the dataloader.
    """
    if len(datasets) == 0:
        raise ValueError("No datasets were provided for LLPR fitting.")
    collate_fn = CollateFn(
        target_keys=list(model.dataset_info.targets.keys()),
        callables=[
            get_system_with_neighbor_lists_transform(
                get_requested_neighbor_lists(model)
            )
        ],
    )

    dataloaders = []
    for dataset in datasets:
        if len(dataset) < batch_size:
            raise ValueError(
                f"A dataset has fewer samples ({len(dataset)}) than the batch size "
                f"({batch_size}). Please reduce the batch size."
            )
        sampler = (
            _NoPadDistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
            )
            if is_distributed
            else None
        )
        dataloaders.append(
            DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                sampler=sampler,
                drop_last=False,
                collate_fn=collate_fn,
            )
        )
    # not shuffled, so that the results do not depend on the number of processes
    return CombinedDataLoader(dataloaders, shuffle=False)


class _NoPadDistributedSampler(torch.utils.data.Sampler[int]):
    """Splits a dataset between processes without padding it, so that each sample
    is seen exactly once.

    :param dataset: the dataset.
    :param num_replicas: the number of processes.
    :param rank: the rank of this process.
    """

    def __init__(
        self, dataset: torch.utils.data.Dataset, num_replicas: int, rank: int
    ) -> None:
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self) -> int:
        n = len(self.dataset)
        return (n - self.rank + self.num_replicas - 1) // self.num_replicas
