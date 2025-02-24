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


class CentralTokensHead(torch.nn.Module):
    def __init__(self, head):
        super(CentralTokensHead, self).__init__()
        self.head = head

    def forward(self, central_tokens: torch.Tensor):
        return self.head(central_tokens)


class MessagesBondsHead(torch.nn.Module):
    def __init__(self, head):
        super(MessagesBondsHead, self).__init__()
        self.head = head

    def forward(
        self,
        messages: torch.Tensor,
        mask: torch.Tensor,
        multipliers: torch.Tensor,
    ):
        predictions = self.head(messages)
        mask_expanded = mask[..., None].repeat(1, 1, predictions.shape[2])
        predictions = torch.where(mask_expanded, 0.0, predictions)
        predictions = predictions * multipliers[:, :, None]
        result = predictions.sum(dim=1)
        return result
