import logging
import warnings
from pathlib import Path
from typing import List, Union

import torch
from metatensor.learn.data import DataLoader

from ...utils.composition import calculate_composition_weights
from ...utils.data import CombinedDataLoader, Dataset, collate_fn, get_all_targets
from ...utils.data.extract_targets import get_targets_dict
from ...utils.evaluate_model import evaluate_model
from ...utils.external_naming import to_external_name
from ...utils.logging import MetricLogger
from ...utils.loss import TensorMapDictLoss
from ...utils.metrics import RMSEAccumulator
from ...utils.per_atom import average_by_num_atoms
from .model import SoapBpnn


logger = logging.getLogger(__name__)


# Filter out the second derivative and device warnings from rascaline-torch
warnings.filterwarnings("ignore", category=UserWarning, message="second derivative")
warnings.filterwarnings(
    "ignore", category=UserWarning, message="Systems data is on device"
)


class Trainer:
    def __init__(self, train_hypers):
        self.hypers = train_hypers

    def train(
        self,
        model: SoapBpnn,
        devices: List[torch.device],
        train_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        validation_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        checkpoints_dir: str,
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
                atomic_types = []
                num_species = len(cur_weight_dict)
                fixed_weights = torch.zeros(num_species, dtype=dtype, device=device)

                for ii, (key, weight) in enumerate(cur_weight_dict.items()):
                    atomic_types.append(key)
                    fixed_weights[ii] = weight

                if not set(atomic_types) == set(model.dataset_info.atomic_types):
                    raise ValueError(
                        "Supplied atomic types are not present in the dataset."
                    )
                model.set_composition_weights(target_name, fixed_weights, atomic_types)

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
                composition_weights, atomic_types = calculate_composition_weights(
                    train_datasets_with_target, target_name
                )
                model.set_composition_weights(
                    target_name, composition_weights, atomic_types
                )

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

        # Create dataloader for the validation datasets:
        validation_dataloaders = []
        for dataset in validation_datasets:
            validation_dataloaders.append(
                DataLoader(
                    dataset=dataset,
                    batch_size=self.hypers["batch_size"],
                    shuffle=False,
                    collate_fn=collate_fn,
                )
            )
        validation_dataloader = CombinedDataLoader(
            validation_dataloaders, shuffle=False
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
            model.parameters(), lr=self.hypers["learning_rate"]
        )

        # Create a scheduler:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.hypers["scheduler_factor"],
            patience=self.hypers["scheduler_patience"],
        )

        # counters for early stopping:
        best_validation_loss = float("inf")
        epochs_without_improvement = 0

        # per-atom targets:
        per_structure_targets = self.hypers["per_structure_targets"]

        # Train the model:
        logger.info("Starting training")
        for epoch in range(self.hypers["num_epochs"]):
            train_rmse_calculator = RMSEAccumulator()
            validation_rmse_calculator = RMSEAccumulator()

            train_loss = 0.0
            for batch in train_dataloader:
                optimizer.zero_grad()

                systems, targets = batch
                systems = [system.to(device=device) for system in systems]
                targets = {
                    key: value.to(device=device) for key, value in targets.items()
                }
                predictions = evaluate_model(
                    model,
                    systems,
                    {key: training_targets[key] for key in targets.keys()},
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
            finalized_train_info = train_rmse_calculator.finalize(
                not_per_atom=["positions_gradients"] + per_structure_targets
            )

            validation_loss = 0.0
            for batch in validation_dataloader:
                systems, targets = batch
                systems = [system.to(device=device) for system in systems]
                targets = {
                    key: value.to(device=device) for key, value in targets.items()
                }
                predictions = evaluate_model(
                    model,
                    systems,
                    {key: training_targets[key] for key in targets.keys()},
                    is_training=False,
                )

                # average by the number of atoms
                predictions = average_by_num_atoms(
                    predictions, systems, per_structure_targets
                )
                targets = average_by_num_atoms(targets, systems, per_structure_targets)

                validation_loss_batch = loss_fn(predictions, targets)
                validation_loss += validation_loss_batch.item()
                validation_rmse_calculator.update(predictions, targets)
            finalized_validation_info = validation_rmse_calculator.finalize(
                not_per_atom=["positions_gradients"] + per_structure_targets
            )

            lr_scheduler.step(validation_loss)

            # Now we log the information:
            finalized_train_info = {"loss": train_loss, **finalized_train_info}
            finalized_validation_info = {
                "loss": validation_loss,
                **finalized_validation_info,
            }

            if epoch == 0:
                metric_logger = MetricLogger(
                    logobj=logger,
                    model_outputs=model.outputs,
                    initial_metrics=[finalized_train_info, finalized_validation_info],
                    names=["train", "validation"],
                )
            if epoch % self.hypers["log_interval"] == 0:
                metric_logger.log(
                    metrics=[finalized_train_info, finalized_validation_info],
                    epoch=epoch,
                )

            if epoch % self.hypers["checkpoint_interval"] == 0:
                model.save_checkpoint(Path(checkpoints_dir) / f"model_{epoch}.ckpt")

            # early stopping criterion:
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.hypers["early_stopping_patience"]:
                    logger.info(
                        "Early stopping criterion reached after "
                        f"{self.hypers['early_stopping_patience']} epochs "
                        "without improvement."
                    )
                    break
