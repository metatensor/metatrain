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
