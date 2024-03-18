import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from metatensor.learn.data import DataLoader
from metatensor.learn.data.dataset import _BaseDataset
from metatensor.torch.atomistic import ModelCapabilities

from ...utils.composition import calculate_composition_weights
from ...utils.compute_loss import compute_model_loss
from ...utils.data import (
    check_datasets,
    collate_fn,
    combine_dataloaders,
    get_all_targets,
)
from ...utils.extract_targets import get_outputs_dict
from ...utils.info import finalize_aggregated_info, update_aggregated_info
from ...utils.io import is_exported, load, save
from ...utils.logging import MetricLogger
from ...utils.loss import TensorMapDictLoss
from ...utils.merge_capabilities import merge_capabilities
from ...utils.neighbors_lists import get_system_with_neighbors_lists
from .model import DEFAULT_HYPERS, Model
from .utils.normalize import (
    get_average_number_of_atoms,
    get_average_number_of_neighbors,
)


logger = logging.getLogger(__name__)


def train(
    train_datasets: List[Union[_BaseDataset, torch.utils.data.Subset]],
    validation_datasets: List[Union[_BaseDataset, torch.utils.data.Subset]],
    requested_capabilities: ModelCapabilities,
    devices: List[torch.device],
    hypers: Dict = DEFAULT_HYPERS,
    continue_from: Optional[str] = None,
    output_dir: str = ".",
):
    if continue_from is None:
        model = Model(
            capabilities=requested_capabilities,
            hypers=hypers["model"],
        )
        new_capabilities = requested_capabilities
    else:
        model = load(continue_from)
        if is_exported(model):
            raise ValueError("model is already exported and can't be used for continue")

        filtered_new_dict = {k: v for k, v in hypers["model"].items() if k != "restart"}
        filtered_old_dict = {k: v for k, v in model.hypers.items() if k != "restart"}
        if filtered_new_dict != filtered_old_dict:
            logger.warning(
                "The hyperparameters of the model have changed since the last "
                "training run. The new hyperparameters will be discarded."
            )
        # merge the model's capabilities with the requested capabilities
        merged_capabilities, new_capabilities = merge_capabilities(
            model.capabilities, requested_capabilities
        )
        model.capabilities = merged_capabilities
        # make the new model capable of handling the new outputs
        for output_name in new_capabilities.outputs.keys():
            model.add_output(output_name)

    model_capabilities = model.capabilities

    # Perform canonical checks on the datasets:
    logger.info("Checking datasets for consistency")
    check_datasets(
        train_datasets,
        validation_datasets,
        model_capabilities,
    )

    # Calculating the neighbors lists for the training and validation datasets:
    logger.info("Calculating neighbors lists for the datasets")
    requested_neighbor_lists = model.requested_neighbors_lists()
    for dataset in train_datasets + validation_datasets:
        for i in range(len(dataset)):
            system = dataset[i].system
            # The following line attached the neighbors lists to the system,
            # and doesn't require to reassign the system to the dataset:
            _ = get_system_with_neighbors_lists(system, requested_neighbor_lists)

    # Calculate the average number of atoms and neighbors in the training datasets:
    average_number_of_atoms = get_average_number_of_atoms(train_datasets)
    average_number_of_neighbors = get_average_number_of_neighbors(train_datasets)

    # Given that currently multiple datasets are not supported, we can assume that:
    average_number_of_atoms = average_number_of_atoms[0]
    average_number_of_neighbors = average_number_of_neighbors[0]

    # Set the normalization factor for the basis functions of the model:

    # Set the normalization factors for the model:
    model.set_normalization_factor(average_number_of_atoms)
    model.set_basis_normalization_factor(average_number_of_neighbors)

    if len(devices) > 1:
        raise ValueError("The alchemical model does not support multiple devices.")
    if devices[0].type == "mps":
        # due to sphericart
        raise ValueError("The alchemical model does not support MPS devices.")
    device = devices[0]
    logger.info(f"Training on device {device}")
    model.to(device)

    # Calculate and set the composition weights for all targets:
    for target_name in new_capabilities.outputs.keys():
        # TODO: warn in the documentation that capabilities that are already
        # present in the model won't recalculate the composition weights
        # find the datasets that contain the target:
        train_datasets_with_target = []
        for dataset in train_datasets:
            if target_name in get_all_targets(dataset):
                train_datasets_with_target.append(dataset)
        if len(train_datasets_with_target) == 0:
            raise ValueError(
                f"Target {target_name} in the model's new capabilities is not "
                "present in any of the training datasets."
            )
        composition_weights, species = calculate_composition_weights(
            train_datasets_with_target, target_name
        )
        model.set_composition_weights(composition_weights.unsqueeze(0), species)

    hypers_training = hypers["training"]

    logger.info("Setting up data loaders")

    # Create dataloader for the training datasets:
    train_dataloaders = []
    for dataset in train_datasets:
        train_dataloaders.append(
            DataLoader(
                dataset=dataset,
                batch_size=hypers_training["batch_size"],
                shuffle=True,
                collate_fn=collate_fn,
            )
        )
    train_dataloader = combine_dataloaders(train_dataloaders, shuffle=True)

    # Create dataloader for the validation datasets:
    validation_dataloaders = []
    for dataset in validation_datasets:
        validation_dataloaders.append(
            DataLoader(
                dataset=dataset,
                batch_size=hypers_training["batch_size"],
                shuffle=False,
                collate_fn=collate_fn,
            )
        )
    validation_dataloader = combine_dataloaders(validation_dataloaders, shuffle=False)

    # Create dataloader for the validation datasets:
    validation_dataloaders = []
    for dataset in validation_datasets:
        validation_dataloaders.append(
            DataLoader(
                dataset=dataset,
                batch_size=hypers_training["batch_size"],
                shuffle=False,
                collate_fn=collate_fn,
            )
        )
    validation_dataloader = combine_dataloaders(validation_dataloaders, shuffle=False)

    # Extract all the possible outputs and their gradients from the training set:
    outputs_dict = get_outputs_dict(train_datasets)
    for output_name in outputs_dict.keys():
        if output_name not in model_capabilities.outputs:
            raise ValueError(
                f"Output {output_name} is not in the model's capabilities."
            )

    # Create a loss weight dict:
    loss_weights_dict = {}
    for output_name, value_or_gradient_list in outputs_dict.items():
        loss_weights_dict[output_name] = {
            value_or_gradient: 1.0 for value_or_gradient in value_or_gradient_list
        }

    # Create a loss function:
    loss_fn = TensorMapDictLoss(loss_weights_dict)

    # Create an optimizer:
    optimizer = torch.optim.Adam(
        model.parameters(), lr=hypers_training["learning_rate"]
    )

    # counters for early stopping:
    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    # Train the model:
    logger.info("Starting training")
    for epoch in range(hypers_training["num_epochs"]):
        # aggregated information holders:
        aggregated_train_info: Dict[str, Tuple[float, int]] = {}
        aggregated_validation_info: Dict[str, Tuple[float, int]] = {}

        train_loss = 0.0
        for batch in train_dataloader:
            optimizer.zero_grad()

            systems, targets = batch
            assert len(systems[0].known_neighbors_lists()) > 0
            loss, info = compute_model_loss(
                loss_fn, model, systems, targets, hypers_training["per_atom_targets"]
            )

            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            aggregated_train_info = update_aggregated_info(aggregated_train_info, info)
        finalized_train_info = finalize_aggregated_info(aggregated_train_info)

        validation_loss = 0.0
        for batch in validation_dataloader:
            systems, targets = batch
            # TODO: specify that the model is not training here to save some autograd

            loss, info = compute_model_loss(
                loss_fn, model, systems, targets, hypers_training["per_atom_targets"]
            )

            validation_loss += loss.item()
            aggregated_validation_info = update_aggregated_info(
                aggregated_validation_info, info
            )
        finalized_validation_info = finalize_aggregated_info(aggregated_validation_info)

        # Now we log the information:
        if epoch == 0:
            metric_logger = MetricLogger(
                model_capabilities,
                train_loss,
                validation_loss,
                finalized_train_info,
                finalized_validation_info,
            )
        if epoch % hypers_training["log_interval"] == 0:
            metric_logger.log(
                epoch,
                train_loss,
                validation_loss,
                finalized_train_info,
                finalized_validation_info,
            )

        if epoch % hypers_training["checkpoint_interval"] == 0:
            save(
                model,
                Path(output_dir) / f"model_{epoch}.pt",
            )

        # early stopping criterion:
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= 50:
                logger.info(
                    "Early stopping criterion reached after 50 "
                    "epochs without improvement."
                )
                break

    return model
