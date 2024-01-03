import logging
from pathlib import Path

import torch

from ..utils.composition import calculate_composition_weights
from ..utils.data import collate_fn
from ..utils.model_io import save_model
from .model import DEFAULT_HYPERS, Model


logger = logging.getLogger(__name__)


def loss_function(predicted, target):
    return torch.sum((predicted.block().values - target.block().values) ** 2)


def train(train_dataset, hypers=DEFAULT_HYPERS, output_dir="."):
    # Calculate and set the composition weights:

    model = Model(
        all_species=train_dataset.all_species,
        hypers=hypers["model"],
    )

    if len(train_dataset.targets) > 1:
        raise ValueError(
            f"`train_dataset` contains {len(train_dataset.targets)} targets but we "
            "currently only support a single target value!"
        )
    else:
        target = list(train_dataset.targets.keys())[0]

    composition_weights = calculate_composition_weights(train_dataset, target)
    model.set_composition_weights(composition_weights)

    hypers_training = hypers["training"]

    # Create a dataloader for the training dataset:
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=hypers_training["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
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
            predicted = model(structures)
            loss = loss_function(predicted["energy"], targets["U0"])
            loss.backward()
            optimizer.step()

    return model
