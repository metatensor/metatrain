import logging
import math
from typing import Tuple, Union

import torch
from torch.optim.lr_scheduler import LambdaLR

from ..documentation import TrainerHypers
from ..model import PET


def get_optimizer(model: PET, hypers: TrainerHypers) -> torch.optim.Optimizer:
    """
    Get the optimizer based on the hyperparameters.

    :param model: The model to optimize.
    :param hypers: The training hyperparameters.
    :return: The optimizer.
    """
    if hypers["weight_decay"] is None:
        weight_decay = 0.0
    else:
        weight_decay = hypers["weight_decay"]
    lr = hypers.get("learning_rate", 1e-4)
    if hypers["optimizer"].lower() == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    elif hypers["optimizer"].lower() == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    elif hypers["optimizer"].lower() == "muon":
        logging.warning(
            "Using the Muon optimizer with auxiliary AdamW for non-matrix "
            "parameters. This feature is experimental and so far not well tested. "
            "Please use it with caution or set the optimizer to Adam or AdamW in the "
            "options.yaml."
        )
        # Separate parameters into Muon and Adam groups.
        # By design, Muon should only be used for the matrix-type parameters
        # (i. e. those with ndim >= 2), and only for optimizing the hidden
        # layers of the model (in our case, the GNN layers). All other parameters
        # including biases, embeddings, and readout layers (heads) should be
        # optimized with Adam or AdamW.
        muon_params = []
        adam_params = []
        for n, p in model.named_parameters():
            if p.ndim >= 2 and "gnn_layers" in n and "neighbor_embedder" not in n:
                muon_params.append(p)
            else:
                adam_params.append(p)
        adam_group = dict(params=adam_params, use_muon=False)
        muon_group = dict(params=muon_params, use_muon=True)
        optimizer = MuonWithAuxAdamW(
            [muon_group, adam_group],
            lr=lr,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(
            f"Unknown optimizer: {hypers['optimizer']}. Please choose Adam, "
            f"AdamW or Muon."
        )

    return optimizer


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    train_hypers: TrainerHypers,
    steps_per_epoch: int,
) -> LambdaLR:
    """
    Get a CosineAnnealing learning-rate scheduler with warmup

    :param optimizer: The optimizer for which to create the scheduler.
    :param train_hypers: The training hyperparameters.
    :param steps_per_epoch: The number of steps per epoch.
    :return: The learning rate scheduler.
    """
    total_steps = train_hypers["num_epochs"] * steps_per_epoch
    warmup_steps = int(train_hypers["warmup_fraction"] * total_steps)
    min_lr_ratio = 0.0  # hardcoded for now, could be made configurable in the future

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine decay
            progress = (current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    return scheduler


class MuonWithAuxAdamW(torch.optim.Optimizer):
    def __init__(
        self,
        param_groups,
        lr: Union[float, torch.Tensor] = 0.001,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
        eps: float = 1e-10,
        betas: Tuple[float, float] = (0.9, 0.95),
    ):
        # Set defaults that will be merged into param_groups
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eps=eps,
            betas=betas,
        )

        # Initialize base optimizer first (this merges defaults into param_groups)
        super().__init__(param_groups, defaults)

        # Now create the internal optimizers using the fully initialized param_groups
        for group in self.param_groups:
            assert "use_muon" in group
            params = group["params"]
            if group["use_muon"]:
                self.muon_optimizer = torch.optim.Muon(
                    params,
                    lr=group["lr"],
                    momentum=group["momentum"],
                )
            else:
                self.adamw_optimizer = torch.optim.AdamW(
                    params,
                    lr=group["lr"],
                    betas=group["betas"],
                    eps=group["eps"],
                    weight_decay=group["weight_decay"],
                )

    @torch.no_grad()
    def step(self, closure=None):
        self.muon_optimizer.step()
        self.adamw_optimizer.step()

    def zero_grad(self, set_to_none: bool = True):
        self.muon_optimizer.zero_grad(set_to_none=set_to_none)
        self.adamw_optimizer.zero_grad(set_to_none=set_to_none)

    def load_state_dict(self, state_dict):
        self.muon_optimizer.load_state_dict(state_dict["muon_optimizer"])
        self.adamw_optimizer.load_state_dict(state_dict["adamw_optimizer"])

    def state_dict(self):
        return {
            "muon_optimizer": self.muon_optimizer.state_dict(),
            "adamw_optimizer": self.adamw_optimizer.state_dict(),
        }
