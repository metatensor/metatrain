import logging

import torch

from ..utils.data import collate_fn


def loss_function(predicted, target):
    return torch.sum((predicted.block().values - target.block().values) ** 2)


def train(model, train_dataset, hypers):
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
            logging.info(f"Epoch {epoch}")
        if epoch % hypers["checkpoint_interval"] == 0:
            torch.save(model.state_dict(), f"model-{epoch}.pt")
        for batch in train_dataloader:
            optimizer.zero_grad()
            structures, targets = batch
            predicted = model(structures)
            loss = loss_function(predicted["energy"], targets["U0"])
            loss.backward()
            optimizer.step()

    # Save the model:
    torch.save(model.state_dict(), "model_final.pt")
