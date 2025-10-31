import copy
import logging
import math
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import ModelMetadata, ModelOutput, NeighborListOptions, System
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler

from .abc import ModelInterface, TrainerInterface
from .additive import get_remove_additive_transform
from .augmentation import RotationalAugmenter
from .data import (
    CollateFn,
    CombinedDataLoader,
    Dataset,
    get_num_workers,
    unpack_batch,
    validate_num_workers,
)
from .data.dataset import DatasetInfo
from .distributed.distributed_data_parallel import DistributedDataParallel
from .distributed.slurm import DistributedEnvironment
from .evaluate_model import evaluate_model
from .io import check_file_extension
from .logging import ROOT_LOGGER, MetricLogger
from .loss import LossAggregator
from .metrics import MAEAccumulator, RMSEAccumulator, get_selected_metric
from .neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists_transform,
)
from .per_atom import average_by_num_atoms
from .scaler import get_remove_scale_transform
from .transfer import batch_to


class MLIPModel(ModelInterface):
    __checkpoint_version__ = 1
    __supported_devices__ = ["cuda", "cpu"]
    __supported_dtypes__ = [torch.float64, torch.float32]
    __default_metadata__ = ModelMetadata()

    def __init__(self, hypers: Dict[str, Any], dataset_info: DatasetInfo) -> None:
        super().__init__(hypers, dataset_info, self.__default_metadata__)

        if len(dataset_info.targets) > 1:
            raise ValueError(
                "MLIPModel only supports datasets with a single target. "
                f"Found {len(dataset_info.targets)} targets."
            )
        self.target_name = dataset_info.targets.keys()[0]
        if dataset_info.targets[self.target_name].quantity != "energy":
            raise ValueError(
                "MLIPModel only supports datasets with an energy as target quantity. "
                f"Found '{dataset_info.targets[self.target_name].quantity}'."
            )
        if not dataset_info.targets[self.target_name].is_scalar:
            raise ValueError(
                "MLIPModel only supports datasets with a scalar target. "
                "Found a non-scalar target."
            )
        if dataset_info.targets[self.target_name].per_atom:
            raise ValueError(
                "MLIPModel only supports datasets with a total energy target. "
                "Found a per-atom target."
            )
        if (dataset_info.targets[self.target_name].layout.block().properties) > 1:
            raise ValueError(
                "MLIPModel only supports datasets with a single sub-target. "
                "Found "
                f"{dataset_info.targets[self.target_name].layout.block().properties} "
                "sub-targets."
            )

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        if len(outputs) > 1:
            raise ValueError(
                "MLIPModel only supports a single output. "
                f"Found {len(outputs)} outputs."
            )
        if self.target_name not in outputs:
            raise ValueError(
                f"MLIPModel only supports the '{self.target_name}' output. "
                f"Found outputs: {list(outputs.keys())}."
            )
        if selected_atoms is not None:
            raise ValueError(
                "MLIPModel does not support the 'selected_atoms' argument."
            )

        positions = []
        centers = []
        neighbors = []
        species = []
        cell_shifts = []
        cells = []
        node_counter = 0
        for system in systems:
            positions.append(system.positions)
            species.append(system.types)
            assert len(system.known_neighbor_lists()) == 1, "no neighbor list found"
            neighbor_list = system.get_neighbor_list(self.nl_options)
            nl_values = neighbor_list.samples.values
            centers.append(nl_values[:, 0] + node_counter)
            neighbors.append(nl_values[:, 1] + node_counter)
            cell_shifts.append(nl_values[:, 2:])
            cells.append(system.cell)
            node_counter += len(system.positions)

        positions = torch.cat(positions)
        centers = torch.cat(centers)
        neighbors = torch.cat(neighbors)
        species = torch.cat(species)
        cells = torch.stack(cells)
        cell_shifts = torch.cat(cell_shifts)
        system_indices = torch.concatenate(
            [
                torch.full(
                    (len(system),),
                    i_system,
                    device=positions.device,
                )
                for i_system, system in enumerate(systems)
            ],
        )

        # somehow the backward of this operation is very slow at evaluation,
        # where there is only one cell, therefore we simplify the calculation
        # for that case
        if len(cells) == 1:
            cell_contributions = cell_shifts.to(cells.dtype) @ cells[0]
        else:
            cell_contributions = torch.einsum(
                "ab, abc -> ac",
                cell_shifts.to(cells.dtype),
                cells[system_indices[centers]],
            )
        edge_vectors = positions[neighbors] - positions[centers] + cell_contributions

        energy_as_tensor = self.compute_energy(
            edge_vectors, species, centers, neighbors, system_indices
        )

        energy_as_tensor_map = TensorMap(
            keys=Labels(
                ["_"],
                torch.tensor([[0]], dtype=torch.int64, device=energy_as_tensor.device),
            ),
            blocks=[
                TensorBlock(
                    values=energy_as_tensor.unsqueeze(-1),
                    samples=Labels(
                        names=["structure"],
                        values=torch.arange(
                            len(energy_as_tensor),
                            device=energy_as_tensor.device,
                        ).unsqueeze(-1),
                    ),
                    components=[],
                    properties=Labels(
                        names=["energy"],
                        values=torch.tensor(
                            [[0]], dtype=torch.int64, device=energy_as_tensor.device
                        ),
                    ),
                )
            ],
        )

        return {self.target_info.name: energy_as_tensor_map}

    def request_neighbor_list(self, cutoff) -> None:
        self.nl_options = NeighborListOptions(
            cutoff=cutoff,
            full=True,
            strict=True,
        )

        def requested_neighbor_lists():
            return [self.nl_options]

        self.requested_neighbor_lists = requested_neighbor_lists

    @abstractmethod
    def compute_energy(
        self,
        edge_vectors: torch.Tensor,
        species: torch.Tensor,
        centers: torch.Tensor,
        neighbors: torch.Tensor,
        system_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the total energy given the edge vectors and other information.

        :param edge_vectors: Tensor of shape (N_edges, 3) containing the vectors
            between neighboring atoms.
        :param species: Tensor of shape (N_atoms,) containing the atomic species
            indices.
        :param centers: Tensor of shape (N_edges,) containing the indices of the
            center atoms for each edge.
        :param neighbors: Tensor of shape (N_edges,) containing the indices of the
            neighbor atoms for each edge.
        :param system_indices: Tensor of shape (N_atoms,) containing the indices
            of the systems each atom belongs to.

        :return: Tensor of shape (N_systems,) containing the total energy for each
            system.
        """


def get_mlip_scheduler(
    optimizer: torch.optim.Optimizer, train_hypers: Dict[str, Any], steps_per_epoch: int
) -> LambdaLR:
    """
    Get a CosineAnnealing learning-rate scheduler with warmup for MLIP trainers.

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


class MLIPTrainer(TrainerInterface):
    """
    Base trainer class for MLIP-only architectures.

    This class provides common training logic for models that only predict energies
    and forces. Derived classes can customize behavior by implementing abstract methods.
    """

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

    @abstractmethod
    def use_rotational_augmentation(self) -> bool:
        """
        Specify whether the trainer should use rotational augmentation.

        :return: True if rotational augmentation should be used, False otherwise.
        """

    def train(
        self,
        model: MLIPModel,
        dtype: torch.dtype,
        devices: List[torch.device],
        train_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        val_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        checkpoint_dir: str,
    ) -> None:
        """
        Train the MLIP model.

        :param model: The MLIP model to train.
        :param dtype: The dtype to use for training.
        :param devices: The devices to use for training.
        :param train_datasets: The training datasets.
        :param val_datasets: The validation datasets.
        :param checkpoint_dir: The directory to save checkpoints.
        """
        assert dtype in model.__supported_dtypes__

        is_distributed = self.hypers["distributed"]

        if is_distributed:
            if len(devices) > 1:
                raise ValueError(
                    "Requested distributed training with the `multi-gpu` device. "
                    "If you want to run distributed training with this MLIP model, "
                    "please set `device` to cuda."
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

        # Move the model to the device and dtype:
        model.to(device=device, dtype=dtype)
        # The additive models are always in float64 (to avoid numerical errors in
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

        # Create collate function(s):
        dataset_info = model.dataset_info
        train_targets = dataset_info.targets
        extra_data_info = dataset_info.extra_data
        requested_neighbor_lists = get_requested_neighbor_lists(model)

        # Check if rotational augmentation should be used
        use_augmentation = self.use_rotational_augmentation()

        if use_augmentation:
            # Create separate collate functions for train and validation
            rotational_augmenter = RotationalAugmenter(
                target_info_dict=train_targets, extra_data_info_dict=extra_data_info
            )
            collate_fn_train = CollateFn(
                target_keys=list(train_targets.keys()),
                callables=[
                    rotational_augmenter.apply_random_augmentations,
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
        else:
            # Use same collate function for both train and validation (no augmentation)
            collate_fn_train = CollateFn(
                target_keys=list(train_targets.keys()),
                callables=[
                    get_system_with_neighbor_lists_transform(requested_neighbor_lists),
                    get_remove_additive_transform(additive_models, train_targets),
                    get_remove_scale_transform(scaler),
                ],
            )
            collate_fn_val = collate_fn_train

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

        # Extract all the possible outputs and their gradients:
        train_targets = (model.module if is_distributed else model).dataset_info.targets
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

        # Create an optimizer:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.hypers["learning_rate"]
        )
        if self.optimizer_state_dict is not None:
            # try to load the optimizer state dict, but this is only possible
            # if there are no new targets in the model (new parameters)
            if not (model.module if is_distributed else model).has_new_targets:
                optimizer.load_state_dict(self.optimizer_state_dict)

        # Create a learning rate scheduler
        lr_scheduler = get_mlip_scheduler(optimizer, self.hypers, len(train_dataloader))

        if self.scheduler_state_dict is not None:
            # same as the optimizer, try to load the scheduler state dict
            if not (model.module if is_distributed else model).has_new_targets:
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

                if is_distributed:
                    # make sure all parameters contribute to the gradient calculation
                    # to make torch DDP happy
                    for param in model.parameters():
                        train_loss_batch += 0.0 * param.sum()

                train_loss_batch.backward()
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
            finalized_train_info = {"loss": train_loss, **finalized_train_info}
            finalized_val_info = {"loss": val_loss, **finalized_val_info}

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
        """Save a checkpoint of the model and trainer state."""
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
        hypers: Dict[str, Any],
        context: Literal["restart", "finetune"],  # not used at the moment
    ) -> "MLIPTrainer":
        """Load trainer state from a checkpoint."""
        trainer = cls(hypers)
        trainer.optimizer_state_dict = checkpoint["optimizer_state_dict"]
        trainer.scheduler_state_dict = checkpoint["scheduler_state_dict"]
        trainer.epoch = checkpoint["epoch"]
        trainer.best_epoch = checkpoint["best_epoch"]
        trainer.best_metric = checkpoint["best_metric"]
        trainer.best_model_state_dict = checkpoint["best_model_state_dict"]
        trainer.best_optimizer_state_dict = checkpoint["best_optimizer_state_dict"]

        return trainer
