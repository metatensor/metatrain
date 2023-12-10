import logging

import torch

from ..utils.composition import calculate_composition_weights
from ..utils.data import collate_fn
from ..utils.model_io import save_model
from .model import DEAFAULT_HYPERS


DEFAULT_TRAINING_HYPERS = DEAFAULT_HYPERS["training"]

logger = logging.getLogger(__name__)


def loss_function(predicted, target):
    return torch.sum((predicted.block().values - target.block().values) ** 2)


def train(model, train_dataset, hypers=DEFAULT_TRAINING_HYPERS):
    # Calculate and set the composition weights:
    composition_weights = calculate_composition_weights(train_dataset, "U0")
    model.set_composition_weights(composition_weights)

    # Create a dataloader for the training dataset:
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=hypers["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )

    # Create an optimizer:
    optimizer = torch.optim.Adam(model.parameters(), lr=hypers["learning_rate"])

    # Train the model:
    for epoch in range(hypers["num_epochs"]):
        if epoch % hypers["log_interval"] == 0:
            logger.info(f"Epoch {epoch}")
        if epoch % hypers["checkpoint_interval"] == 0:
            save_model(
                model,
                f"model_{epoch}.pt",
            )
        for batch in train_dataloader:
            optimizer.zero_grad()
            structures, targets = batch
            predicted = model(structures)
            loss = loss_function(predicted["energy"], targets["U0"])
            loss.backward()
            optimizer.step()

    # Save the model:
    save_model(model, "model_final.pt")
