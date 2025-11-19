import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Union

import torch
from torch.utils.data import DataLoader

from metatrain.utils.abc import ModelInterface, TrainerInterface
from metatrain.utils.data import (
    CollateFn,
    CombinedDataLoader,
    Dataset,
    unpack_batch,
)
from metatrain.utils.io import check_file_extension, model_from_checkpoint
from metatrain.utils.logging import ROOT_LOGGER, MetricLogger
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)

from . import checkpoints
from .documentation import TrainerHypers
from .model import Classifier


class Trainer(TrainerInterface[TrainerHypers]):
    __checkpoint_version__ = 1

    def train(
        self,
        model: Classifier,
        dtype: torch.dtype,
        devices: List[torch.device],
        train_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        val_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        checkpoint_dir: str,
    ) -> None:
        # Load the wrapped model from checkpoint and set it as the wrapped model of the
        # Classifier model:
        if self.hypers["model_checkpoint"] is None:
            raise ValueError(
                "A model checkpoint must be provided to train the Classifier "
                "(model_checkpoint, under training, in the hypers)"
            )
        wrapped_model_checkpoint_path = self.hypers["model_checkpoint"]
        checkpoint = torch.load(
            wrapped_model_checkpoint_path, weights_only=False, map_location="cpu"
        )
        wrapped_model = model_from_checkpoint(checkpoint, "export")
        model.set_wrapped_model(wrapped_model)

        device = devices[0]  # this trainer doesn't support multi-GPU training
        # check device and dtype against wrapped model class
        if device.type not in wrapped_model.__class__.__supported_devices__:
            raise ValueError(
                f"Device {device} not supported by the wrapped model. "
                f"Supported devices are {wrapped_model.__class__.__supported_devices__}"
            )
        if dtype not in wrapped_model.__class__.__supported_dtypes__:
            raise ValueError(
                f"dtype {dtype} not supported by the wrapped model. "
                f"Supported dtypes are {wrapped_model.__class__.__supported_dtypes__}"
            )
        logging.info(f"Training on device {device} with dtype {dtype}")

        logging.info("Calculating neighbor lists for the datasets")
        # Calculate the neighbor lists in advance, if needed
        requested_neighbor_lists = get_requested_neighbor_lists(model)
        for dataset in train_datasets + val_datasets:
            for sample in dataset:
                system = sample["system"]
                # The following line attaches the neighbors lists to the system,
                # and doesn't require to reassign the system to the dataset:
                get_system_with_neighbor_lists(system, requested_neighbor_lists)

        # Move the model to the device and dtype:
        model.to(device=device, dtype=dtype)

        # Determine number of classes from the training data (one-hot encoded)
        num_classes = (
            train_datasets[0][0]
            ._asdict()[list(model.dataset_info.targets.keys())[0]]
            .block()
            .values.shape[-1]
        )

        logging.info(f"Number of classes detected: {num_classes}")

        # Get feature size by doing a forward pass on one sample
        import metatensor.torch as mts
        from metatomic.torch import ModelOutput

        sample = train_datasets[0][0]
        system = sample["system"].to(device=device, dtype=dtype)
        with torch.no_grad():
            features_dict = model.model(
                [system],
                {"features": ModelOutput(per_atom=True)},
            )
            averaged_features = mts.mean_over_samples(
                features_dict["features"], sample_names=["atom"]
            )
            feature_size = averaged_features.block().values.shape[-1]

        logging.info(f"Feature size: {feature_size}")

        # Build the MLP
        model.build_mlp(feature_size, num_classes)
        model.to(device=device, dtype=dtype)

        logging.info("Setting up data loaders")

        # Create a collate function:
        targets_keys = list(model.dataset_info.targets.keys())
        collate_fn = CollateFn(target_keys=targets_keys)

        # Create dataloader for the training datasets:
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
                    drop_last=False,
                    collate_fn=collate_fn,
                )
            )
        train_dataloader = CombinedDataLoader(train_dataloaders, shuffle=True)

        # Create dataloader for the validation datasets:
        val_dataloaders = []
        for val_dataset in val_datasets:
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
                    shuffle=False,
                    drop_last=False,
                    collate_fn=collate_fn,
                )
            )
        val_dataloader = CombinedDataLoader(val_dataloaders, shuffle=False)

        # Setup optimizer
        optimizer = torch.optim.Adam(
            model.mlp.parameters(),
            lr=self.hypers["learning_rate"],
        )

        # Log the initial learning rate:
        logging.info(f"Learning rate: {self.hypers['learning_rate']}")

        # Train the model:
        logging.info("Starting training")
        target_name = list(model.dataset_info.targets.keys())[0]

        best_val_loss = float("inf")
        best_epoch = 0
        best_model_state_dict = None

        for epoch in range(self.hypers["num_epochs"]):
            # Training
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch in train_dataloader:
                optimizer.zero_grad()

                systems, targets, _ = unpack_batch(batch)
                systems = [system.to(device=device, dtype=dtype) for system in systems]
                targets = {
                    name: target.to(device=device, dtype=dtype)
                    for name, target in targets.items()
                }

                # Forward pass
                outputs = model(
                    systems,
                    {target_name: ModelOutput(quantity="", unit="", per_atom=False)},
                    None,
                )

                probabilities = outputs[target_name].block().values
                # Get target probabilities (supports both one-hot and soft targets)
                target_probs = targets[target_name].block().values

                # Compute cross-entropy loss with soft targets
                # CE = -sum(target_probs * log(predicted_probs))
                # Add small epsilon to avoid log(0)
                log_probs = torch.log(probabilities + 1e-10)
                loss = -torch.sum(target_probs * log_probs, dim=-1).mean()

                # Backward pass
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                # For accuracy, compare predicted class with target's most likely class
                _, predicted = torch.max(probabilities, 1)
                _, target_class = torch.max(target_probs, 1)
                train_total += target_class.size(0)
                train_correct += (predicted == target_class).sum().item()

            train_loss /= len(train_dataloader)
            train_acc = train_correct / train_total

            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch in val_dataloader:
                    systems, targets, _ = unpack_batch(batch)
                    systems = [
                        system.to(device=device, dtype=dtype) for system in systems
                    ]
                    targets = {
                        name: target.to(device=device, dtype=dtype)
                        for name, target in targets.items()
                    }

                    # Forward pass
                    outputs = model(
                        systems,
                        {
                            target_name: ModelOutput(
                                quantity="", unit="", per_atom=False
                            )
                        },
                        None,
                    )

                    probabilities = outputs[target_name].block().values
                    # Get target probabilities (supports both one-hot and soft targets)
                    target_probs = targets[target_name].block().values

                    # Compute cross-entropy loss with soft targets
                    # CE = -sum(target_probs * log(predicted_probs))
                    log_probs = torch.log(probabilities + 1e-10)
                    loss = -torch.sum(target_probs * log_probs, dim=-1).mean()

                    val_loss += loss.item()
                    # For accuracy, compare predicted with target's most likely class
                    _, predicted = torch.max(probabilities, 1)
                    _, target_class = torch.max(target_probs, 1)
                    val_total += target_class.size(0)
                    val_correct += (predicted == target_class).sum().item()

            val_loss /= len(val_dataloader)
            val_acc = val_correct / val_total

            # Prepare metrics for logging
            finalized_train_info = {
                "loss": train_loss,
                "accuracy": train_acc,
            }
            finalized_val_info = {
                "loss": val_loss,
                "accuracy": val_acc,
            }

            # Initialize metric logger on first epoch
            if epoch == 0:
                metric_logger = MetricLogger(
                    log_obj=ROOT_LOGGER,
                    dataset_info=model.dataset_info,
                    initial_metrics=[finalized_train_info, finalized_val_info],
                    names=["training", "validation"],
                )

            # Log progress
            if epoch % self.hypers["log_interval"] == 0:
                metric_logger.log(
                    metrics=[finalized_train_info, finalized_val_info],
                    epoch=epoch,
                )

            # Save checkpoint
            if epoch % self.hypers["checkpoint_interval"] == 0:
                checkpoint_path = Path(checkpoint_dir) / f"checkpoint_{epoch}.ckpt"
                self.save_checkpoint(model, checkpoint_path)

            # Track best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_model_state_dict = copy.deepcopy(model.state_dict())

        # Load best model
        if best_model_state_dict is not None:
            model.load_state_dict(best_model_state_dict)
            logging.info(
                f"Best model loaded from epoch {best_epoch} "
                f"with validation loss: {best_val_loss:.6f}"
            )

    def save_checkpoint(self, model: ModelInterface, path: Union[str, Path]) -> None:
        checkpoint = model.get_checkpoint()
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
    ) -> "Classifier":
        raise ValueError("Classifier does not allow restarting training")

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
