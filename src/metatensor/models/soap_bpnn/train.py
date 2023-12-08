import logging

import torch
from omegaconf import OmegaConf

from metatensor.models import ARCHITECTURE_CONFIG_PATH

from ..utils.composition import calculate_composition_weights
from ..utils.data import collate_fn
from ..utils.model_io import save_model
from .model import ARCHITECTURE_NAME


DEFAULT_TRAINING_HYPERS = OmegaConf.to_container(
    OmegaConf.load(ARCHITECTURE_CONFIG_PATH / "soap_bpnn.yaml")
)["training"]

logger = logging.getLogger(__name__)


def loss_function(predicted, target):
    return torch.sum((predicted.block().values - target.block().values) ** 2)


def train(model, train_dataset, hypers=DEFAULT_TRAINING_HYPERS):
    model_hypers = hypers["model"]
    training_hypers = hypers["training"]

    # Calculate and set the composition weights:
    composition_weights = calculate_composition_weights(train_dataset, "U0")
    model.set_composition_weights(composition_weights)

    # Create a dataloader for the training dataset:
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=training_hypers["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )

    # Create an optimizer:
    optimizer = torch.optim.Adam(
        model.parameters(), lr=training_hypers["learning_rate"]
    )

    # Train the model:
    for epoch in range(training_hypers["num_epochs"]):
        if epoch % training_hypers["log_interval"] == 0:
            logger.info(f"Epoch {epoch}")
        if epoch % training_hypers["checkpoint_interval"] == 0:
            save_model(
                ARCHITECTURE_NAME,
                model,
                model_hypers,
                model.all_species,
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
    save_model(
        ARCHITECTURE_NAME, model, model_hypers, model.all_species, "model_final.pt"
    )
