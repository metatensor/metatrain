import logging
from pathlib import Path

import torch
from metatensor.torch.atomistic import ModelCapabilities, ModelOutput

from ..utils.composition import calculate_composition_weights
from ..utils.compute_loss import compute_model_loss
from ..utils.data import collate_fn, Dataset, canonical_check_datasets, combine_dataloaders
from ..utils.loss import TensorMapDictLoss
from ..utils.model_io import save_model
from .model import DEFAULT_HYPERS, Model

from typing import Dict, List


logger = logging.getLogger(__name__)


def train(
    train_datasets: List[Dataset],
    validation_datasets: List[Dataset],
    model_capabilities: ModelCapabilities,
    hypers: Dict = DEFAULT_HYPERS,
    output_dir: str = "."
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
    for target_name in model_capabilities.targets:
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
        composition_weights = calculate_composition_weights(train_dataset_with_target, target_name)
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

    #####################################
    I DON'T UNDERSTAND THIS PART
    #####################################

    # Create a loss function:
    loss_fn = TensorMapDictLoss(
        {target_name: {"values": 1.0}},
    )

    # Create an optimizer:
    optimizer = torch.optim.Adam(
        model.parameters(), lr=hypers_training["learning_rate"]
    )

    # Train the model:
    for epoch in range(hypers_training["num_epochs"]):
        if epoch % hypers_training["log_interval"] == 0:
            logger.info(f"Epoch {epoch}")
        if epoch % hypers_training["checkpoint_interval"] == 0:
            save_model(
                model,
                Path(output_dir) / f"model_{epoch}.pt",
            )
        for batch in train_dataloader:
            optimizer.zero_grad()
            structures, targets = batch
            loss = compute_model_loss(loss_fn, model, structures, targets)
            loss.backward()
            optimizer.step()

    return model
