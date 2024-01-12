import logging
from pathlib import Path
from typing import Dict, List

import torch
from metatensor.torch.atomistic import ModelCapabilities

from ..utils.composition import calculate_composition_weights
from ..utils.compute_loss import compute_model_loss
from ..utils.data import (
    Dataset,
    canonical_check_datasets,
    collate_fn,
    combine_dataloaders,
)
from ..utils.loss import TensorMapDictLoss
from ..utils.model_io import save_model
from .model import DEFAULT_HYPERS, Model


logger = logging.getLogger(__name__)


def train(
    train_datasets: List[Dataset],
    validation_datasets: List[Dataset],
    model_capabilities: ModelCapabilities,
    hypers: Dict = DEFAULT_HYPERS,
    output_dir: str = ".",
):
    # Perform canonical checks on the datasets:
    canonical_check_datasets(
        train_datasets,
        validation_datasets,
        model_capabilities,
    )

    # Create the model:
    model = Model(
        capabilities=model_capabilities,
        hypers=hypers["model"],
    )

    # Calculate and set the composition weights for all targets:
    for target_name in model_capabilities.outputs.keys():
        # find the dataset that contains the target:
        train_dataset_with_target = None
        for dataset in train_datasets:
            if target_name in dataset.targets:
                train_dataset_with_target = dataset
                break
        if train_dataset_with_target is None:
            raise ValueError(
                f"Target {target_name} in the model's capabilities is not "
                "present in any of the training datasets."
            )
        composition_weights = calculate_composition_weights(
            train_dataset_with_target, target_name
        )
        model.set_composition_weights(target_name, composition_weights)

    hypers_training = hypers["training"]

    # Create dataloader for the training datasets:
    train_dataloaders = []
    for dataset in train_datasets:
        train_dataloaders.append(
            torch.utils.data.DataLoader(
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
            torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=hypers_training["batch_size"],
                shuffle=False,
                collate_fn=collate_fn,
            )
        )
    validation_dataloader = combine_dataloaders(validation_dataloaders, shuffle=False)

    # Extract all the possible outputs and their gradients from the training set:
    outputs_dict = _get_outputs_dict(train_datasets)
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
    for epoch in range(hypers_training["num_epochs"]):
        train_loss = 0.0
        for batch in train_dataloader:
            optimizer.zero_grad()
            structures, targets = batch
            loss = compute_model_loss(loss_fn, model, structures, targets)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        validation_loss = 0.0
        for batch in validation_dataloader:
            structures, targets = batch
            # TODO: specify that the model is not training here to save some autograd
            loss = compute_model_loss(loss_fn, model, structures, targets)
            validation_loss += loss.item()

        if epoch % hypers_training["log_interval"] == 0:
            logger.info(
                f"Epoch {epoch}, train loss: {train_loss:.4f}, "
                f"validation loss: {validation_loss:.4f}"
            )

        if epoch % hypers_training["checkpoint_interval"] == 0:
            save_model(
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
                    f"Early stopping criterion reached after {epoch} "
                    "epochs without improvement."
                )
                break

    return model


def _get_outputs_dict(datasets: List[Dataset]):
    """
    This is a helper function that extracts all the possible outputs and their gradients
    from a list of datasets.

    :param datasets: A list of datasets.

    :returns: A dictionary mapping output names to a list of "values" (always)
        and possible gradients.
    """

    outputs_dict = {}
    for dataset in datasets:
        sample_batch = next(iter(dataset))
        targets = sample_batch[1]  # this is a dictionary of TensorMaps
        for target_name, target_tmap in targets.items():
            if target_name not in outputs_dict:
                outputs_dict[target_name] = [
                    "values"
                ] + target_tmap.block().gradients_list()

    return outputs_dict
