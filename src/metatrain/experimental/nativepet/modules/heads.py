import torch
from torch import nn


class Head(torch.nn.Module):
    def __init__(self, n_in, n_neurons):
        super(Head, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(n_in, n_neurons),
            torch.nn.SiLU(),
            nn.Linear(n_neurons, n_neurons),
            torch.nn.SiLU(),
        )

    def forward(self, inputs: torch.Tensor):
        return self.nn(inputs)


class CentralTokensLastLayer(torch.nn.Module):
    def __init__(self, last_layer):
        super(CentralTokensLastLayer, self).__init__()
        self.last_layer = last_layer

    def forward(self, central_tokens_features: torch.Tensor):
        return self.last_layer(central_tokens_features)


class MessagesLastLayer(torch.nn.Module):
    def __init__(self, last_layer):
        super(MessagesLastLayer, self).__init__()
        self.last_layer = last_layer

    def forward(
        self,
        messages_features: torch.Tensor,
        mask: torch.Tensor,
        multipliers: torch.Tensor,
    ):
        messages_proceed = messages_features * multipliers[:, :, None]
        messages_proceed[mask] = 0.0
        pooled = messages_proceed.sum(dim=1)

        predictions = self.last_layer(pooled)
        return predictions


class MessagesBondsLastLayer(torch.nn.Module):
    def __init__(self, last_layer):
        super(MessagesBondsLastLayer, self).__init__()
        self.last_layer = last_layer

    def forward(
        self,
        messages_features: torch.Tensor,
        mask: torch.Tensor,
        multipliers: torch.Tensor,
    ):
        predictions = self.last_layer(messages_features)
        mask_expanded = mask[..., None].repeat(1, 1, predictions.shape[2])
        predictions = torch.where(mask_expanded, 0.0, predictions)
        predictions = predictions * multipliers[:, :, None]
        result = predictions.sum(dim=1)
        return result
