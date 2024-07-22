import logging
import warnings
from pathlib import Path
from typing import List, Union

import torch
from metatensor.learn.data import DataLoader

from ...utils.composition import calculate_composition_weights
from ...utils.data import (
    CombinedDataLoader,
    Dataset,
    TargetInfoDict,
    collate_fn,
    get_all_targets,
)
from ...utils.data.extract_targets import get_targets_dict
from ...utils.evaluate_model import evaluate_model
from ...utils.external_naming import to_external_name
from ...utils.logging import MetricLogger
from ...utils.loss import TensorMapDictLoss
from ...utils.metrics import RMSEAccumulator, MAEAccumulator
from ...utils.neighbor_lists import get_system_with_neighbor_lists
from ...utils.per_atom import average_by_num_atoms
from ...utils.scaling import calculate_scaling
from ...utils.io import check_file_extension
from .model import PhACE
import copy


logger = logging.getLogger(__name__)


# Filter out the second derivative and device warnings from rascaline-torch
warnings.filterwarnings("ignore", category=UserWarning, message="second derivative")
warnings.filterwarnings(
    "ignore", category=UserWarning, message="Systems data is on device"
)


class Trainer:
    def __init__(self, train_hypers):
        self.hypers = train_hypers
        self.optimizer_state_dict = None
        self.scheduler_state_dict = None
        self.epoch = None

    def train(
        self,
        model: PhACE,
        dtype: torch.dtype,
        devices: List[torch.device],
        train_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        val_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        checkpoint_dir: str,
    ):
        dtype = train_datasets[0][0]["system"].positions.dtype

        # only one device, as we don't support multi-gpu for now
        assert len(devices) == 1
        device = devices[0]

        logger.info(f"training on device {device} with dtype {dtype}")
        model.to(device=device, dtype=dtype)

        # Calculate and set the composition weights for all targets:
        logger.info("Calculating composition weights")
        for target_name in model.new_outputs:
            if "mtm::aux::" in target_name:
                continue
            # TODO: document transfer learning and say that outputs that are already
            # present in the model will keep their composition weights
            if target_name in self.hypers["fixed_composition_weights"].keys():
                logger.info(
                    f"For {target_name}, model will use "
                    "user-supplied composition weights"
                )
                cur_weight_dict = self.hypers["fixed_composition_weights"][target_name]
                atomic_types = set()
                num_species = len(cur_weight_dict)
                fixed_weights = torch.zeros(num_species, dtype=dtype, device=device)

                for ii, (key, weight) in enumerate(cur_weight_dict.items()):
                    atomic_types.add(key)
                    fixed_weights[ii] = weight

                if not set(atomic_types) == model.atomic_types:
                    raise ValueError(
                        "Supplied atomic types are not present in the dataset."
                    )
                model.set_composition_weights(
                    target_name, fixed_weights, list(atomic_types)
                )

            else:
                train_datasets_with_target = []
                for dataset in train_datasets:
                    if target_name in get_all_targets(dataset):
                        train_datasets_with_target.append(dataset)
                if len(train_datasets_with_target) == 0:
                    raise ValueError(
                        f"Target {target_name} in the model's new capabilities is not "
                        "present in any of the training datasets."
                    )
                composition_weights, composition_types = calculate_composition_weights(
                    train_datasets_with_target, target_name
                )
                model.set_composition_weights(
                    target_name, composition_weights, composition_types
                )

        # Calculate NLs:
        logger.info("Calculating neighbors lists for the datasets")
        requested_neighbor_lists = model.requested_neighbor_lists()
        for dataset in train_datasets + val_datasets:
            for i in range(len(dataset)):
                system = dataset[i]["system"]
                # The following line attached the neighbors lists to the system,
                # and doesn't require to reassign the system to the dataset:
                _ = get_system_with_neighbor_lists(system, requested_neighbor_lists)

            logger.info("Setting up data loaders")

        # Create dataloader for the training datasets:
        train_dataloaders = []
        for dataset in train_datasets:
            train_dataloaders.append(
                DataLoader(
                    dataset=dataset,
                    batch_size=self.hypers["batch_size"],
                    shuffle=True,
                    collate_fn=collate_fn,
                )
            )
        train_dataloader = CombinedDataLoader(train_dataloaders, shuffle=True)

        torch.jit.set_fusion_strategy([("DYNAMIC", 0)])
        scripted_model = torch.jit.script(model)

        # scaling:
        # TODO: this will work sub-optimally if the model is restarting with
        # new targets (but it will still work)
        calculate_scaling(scripted_model, train_dataloader, model.dataset_info, device)

        # Create dataloader for the validation datasets:
        val_dataloaders = []
        for dataset in val_datasets:
            val_dataloaders.append(
                DataLoader(
                    dataset=dataset,
                    batch_size=self.hypers["batch_size"],
                    shuffle=False,
                    collate_fn=collate_fn,
                )
            )
        val_dataloader = CombinedDataLoader(
            val_dataloaders, shuffle=False
        )

        # Extract all the possible outputs and their gradients:
        training_targets = get_targets_dict(train_datasets, model.dataset_info)
        outputs_list = []
        for target_name, target_info in training_targets.items():
            outputs_list.append(target_name)
            for gradient_name in target_info.gradients:
                outputs_list.append(f"{target_name}_{gradient_name}_gradients")
        # Create a loss weight dict:
        loss_weights_dict = {}
        for output_name in outputs_list:
            loss_weights_dict[output_name] = (
                self.hypers["loss_weights"][
                    to_external_name(output_name, training_targets)
                ]
                if to_external_name(output_name, training_targets)
                in self.hypers["loss_weights"]
                else 1.0
            )
        loss_weights_dict_external = {
            to_external_name(key, training_targets): value
            for key, value in loss_weights_dict.items()
        }
        logging.info(f"Training with loss weights: {loss_weights_dict_external}")

        # Create a loss function:
        loss_fn = TensorMapDictLoss(loss_weights_dict)

        # Create an optimizer:
        optimizer = torch.optim.Adam(
            scripted_model.parameters(), lr=self.hypers["learning_rate"]
        )
        if self.optimizer_state_dict is not None:
            optimizer.load_state_dict(self.optimizer_state_dict)

        # Create a scheduler:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.hypers["scheduler_factor"],
            patience=self.hypers["scheduler_patience"],
        )
        if self.scheduler_state_dict is not None:
            lr_scheduler.load_state_dict(self.scheduler_state_dict)

        # per-atom targets:
        per_structure_targets = self.hypers["per_structure_targets"]

        # Create an optimizer and a scheduler:
        optimizer = torch.optim.AdamW(scripted_model.parameters(), lr=self.hypers["learning_rate"], amsgrad=True, weight_decay=5e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=self.hypers["scheduler_factor"], patience=self.hypers["scheduler_patience"])

        # per-atom targets:
        per_structure_targets = self.hypers["per_structure_targets"]

        # Train the model:
        logger.info("Starting training")

        best_val_loss = float("inf")
        n_epochs_without_improvement = 0
        for epoch in range(self.hypers["num_epochs"]):
            train_rmse_calculator = RMSEAccumulator()
            train_mae_calculator = MAEAccumulator()
            val_rmse_calculator = RMSEAccumulator()
            val_mae_calculator = MAEAccumulator()

            train_loss = 0.0
            for batch in train_dataloader:
                optimizer.zero_grad()

                systems, targets = batch
                systems = [system.to(device=device, dtype=dtype) for system in systems]
                targets = {
                    key: value.to(device=device, dtype=dtype) for key, value in targets.items()
                }
                predictions = evaluate_model(
                    scripted_model,
                    systems,
                    TargetInfoDict(
                        **{key: training_targets[key] for key in targets.keys()}
                    ),
                    is_training=True,
                )

                # average by the number of atoms
                predictions = average_by_num_atoms(
                    predictions, systems, per_structure_targets
                )
                targets = average_by_num_atoms(targets, systems, per_structure_targets)

                train_loss_batch = loss_fn(predictions, targets)
                train_loss += train_loss_batch.item()
                train_loss_batch.backward()
                optimizer.step()
                train_rmse_calculator.update(predictions, targets)
                train_mae_calculator.update(predictions, targets)

            finalized_train_info = {
                **train_rmse_calculator.finalize(not_per_atom=["positions_gradients"] + per_structure_targets),
                **train_mae_calculator.finalize(),
            }

            val_loss = 0.0
            for batch in val_dataloader:
                systems, targets = batch
                systems = [system.to(device=device, dtype=dtype) for system in systems]
                targets = {
                    key: value.to(device=device, dtype=dtype) for key, value in targets.items()
                }
                predictions = evaluate_model(
                    scripted_model,
                    systems,
                    TargetInfoDict(
                        **{key: training_targets[key] for key in targets.keys()}
                    ),
                    is_training=False,
                )

                # average by the number of atoms
                predictions = average_by_num_atoms(
                    predictions, systems, per_structure_targets
                )
                targets = average_by_num_atoms(targets, systems, per_structure_targets)

                val_loss_batch = loss_fn(predictions, targets)
                val_loss += val_loss_batch.item()
                val_rmse_calculator.update(predictions, targets)
                val_mae_calculator.update(predictions, targets)

            finalized_val_info = {
                **val_rmse_calculator.finalize(not_per_atom=["positions_gradients"] + per_structure_targets),
                **val_mae_calculator.finalize(),
            }

            lr_scheduler.step(val_loss)

            # Now we log the information:
            finalized_train_info = {"loss": train_loss, **finalized_train_info}
            finalized_val_info = {
                "loss": val_loss,
                **finalized_val_info,
            }

            if epoch == 0:
                metric_logger = MetricLogger(
                    log_obj=logger,
                    dataset_info=model.dataset_info,
                    initial_metrics=[finalized_train_info, finalized_val_info],
                    names=["training", "validation"],
                )
            if epoch % self.hypers["log_interval"] == 0:
                metric_logger.log(
                    metrics=[finalized_train_info, finalized_val_info],
                    epoch=epoch,
                )

            # if epoch % self.hypers["checkpoint_interval"] == 0:
            #     model.save_checkpoint(Path(checkpoint_dir) / f"model_{epoch}.ckpt")
            # TODO: how do I make this work given that it's scripted?

            lr_before = optimizer.param_groups[0]["lr"]
            scheduler.step(val_loss)
            lr_after = optimizer.param_groups[0]["lr"]
            if lr_before != lr_after:
                logger.info(f"Learning rate changed from {lr_before} to {lr_after}")
                model.load_state_dict(best_state_dict)
                optimizer.load_state_dict(best_optimizer_state_dict)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr_after
            if lr_after < 1e-6:
                logger.info("Training has converged, stopping")
                break

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                n_epochs_without_improvement = 0
                best_state_dict = copy.deepcopy(scripted_model.state_dict())
                best_optimizer_state_dict = copy.deepcopy(optimizer.state_dict())

            else:
                n_epochs_without_improvement += 1
                if n_epochs_without_improvement >= self.hypers["early_stopping_patience"]:
                    logger.info(
                        f"Stopping early after {n_epochs_without_improvement} epochs without improvement"
                    )
                    break

        model.load_state_dict(best_state_dict)


    def save_checkpoint(self, model, path: Union[str, Path]):
        checkpoint = {
            "model_hypers": {
                "model_hypers": model.hypers,
                "dataset_info": model.dataset_info,
            },
            "model_state_dict": model.state_dict(),
            "train_hypers": self.hypers,
            "epoch": self.epoch,
            "optimizer_state_dict": self.optimizer_state_dict,
            "scheduler_state_dict": self.scheduler_state_dict,
        }
        torch.save(
            checkpoint,
            check_file_extension(path, ".ckpt"),
        )

    @classmethod
    def load_checkpoint(cls, path: Union[str, Path], train_hypers) -> "Trainer":

        # Load the checkpoint
        checkpoint = torch.load(path)
        model_hypers = checkpoint["model_hypers"]
        model_state_dict = checkpoint["model_state_dict"]
        epoch = checkpoint["epoch"]
        optimizer_state_dict = checkpoint["optimizer_state_dict"]
        scheduler_state_dict = checkpoint["scheduler_state_dict"]

        # Create the trainer
        trainer = cls(train_hypers)
        trainer.optimizer_state_dict = optimizer_state_dict
        trainer.scheduler_state_dict = scheduler_state_dict
        trainer.epoch = epoch

        # Create the model
        model = PhACE(**model_hypers)
        model.load_state_dict(model_state_dict)

        return trainer
