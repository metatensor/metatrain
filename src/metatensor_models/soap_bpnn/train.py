import torch
import logging

import metatensor.torch


def loss_function(predicted, target):
    return torch.sum((predicted.block.values - target.block.values)**2)


def train(model, train_dataset, hypers):

    # Create a dataloader for the training dataset:
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=hypers["batch_size"],
        shuffle=True,
    )

    # Create an optimizer:
    optimizer = torch.optim.Adam(model.parameters(), lr=hypers["learning_rate"])

    # Train the model:
    for epoch in range(hypers["epochs"]):
        if epoch % hypers["log_interval"] == 0:
            logging.info(f"Epoch {epoch}")
        if epoch % hypers["checkpoint_interval"] == 0:
            torch.save(model.state_dict(), f"model-{epoch}.pt")
        for batch in train_dataloader:
            optimizer.zero_grad()
            predicted = model(batch)
            loss = loss_function(predicted, batch)
            loss.backward()
            optimizer.step()

    # Save the model:
    torch.save(model.state_dict(), "model_final.pt")
    