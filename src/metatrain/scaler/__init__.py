import logging
from typing import List, Optional, Sequence, Union

import torch
from torch import nn

from metatrain.utils.data import Dataset
from metatrain.utils.data.dataset import Subset
from metatrain.utils.io import load_model

from .documentation import FixedScalerWeights
from .model import Scaler
from .trainer import Trainer


__model__ = Scaler
__trainer__ = Trainer

__authors__ = [
    ("Paolo Pegolo <paolo.pegolo@epfl.ch>", "@ppegolo"),
    ("Joseph W. Abbott <joseph.william.abbott@gmail.com>", "@jwa7"),
]

__maintainers__ = [
    ("Pol Febrer <pol.febrer@epfl.ch>", "@pfebrer"),
]


def train_or_load_scaler(
    scaler: Scaler,
    train_datasets: List[Union[Dataset, Subset]],
    additive_models: List[nn.Module],
    batch_size: int,
    is_distributed: bool,
    fixed_weights: Optional[FixedScalerWeights | str] = None,
    per_structure_targets: Sequence[str] = (),
    trainer_hypers: Optional[dict] = None,
    checkpoint_dir: str = "",
) -> None:
    """
    Train the scaler from data or load pre-trained weights.

    This is the single source of truth for how to set up a scaler
    for use for preprocessing by any architecture.

    :param scaler: The scaler to train or load into
    :param train_datasets: Training datasets
    :param additive_models: Additive models to
        subtract before fitting
    :param batch_size: Batch size for data loading
    :param is_distributed: Whether training is distributed
    :param fixed_weights: Fixed weights dict, or path to a checkpoint
    :param per_structure_targets: Target names that should be treated as
        per-structure quantities and therefore not divided by the number of atoms.
    :param trainer_hypers: Additional hyperparameters for the trainer.
    :param checkpoint_dir: Directory to save the composition model checkpoint
    """
    if isinstance(fixed_weights, str):
        logging.info(f"Loading scaler from {fixed_weights}")
        loaded = load_model(fixed_weights)
        if not isinstance(loaded, Scaler):
            raise ValueError(
                f"The model loaded from {fixed_weights} is a "
                f"{type(loaded).__name__}, not a Scaler."
            )
        if loaded.atomic_types != scaler.atomic_types:
            raise ValueError(
                "Scaler checkpoint atomic types "
                f"({loaded.atomic_types}) do not match the current model's "
                f"atomic types ({scaler.atomic_types})."
            )
        loaded_targets = loaded.dataset_info.targets
        current_targets = scaler.dataset_info.targets
        if set(loaded_targets) != set(current_targets):
            raise ValueError(
                "Composition checkpoint targets "
                f"({sorted(loaded_targets)}) do not match the current model's "
                f"targets ({sorted(current_targets)})."
            )
        for name, target_info in current_targets.items():
            loaded_info = loaded_targets[name]
            if (loaded_info.quantity, loaded_info.unit) != (
                target_info.quantity,
                target_info.unit,
            ):
                raise ValueError(
                    f"Target '{name}' from the scaler checkpoint has "
                    f"quantity '{loaded_info.quantity}' and unit "
                    f"'{loaded_info.unit}', while the current model expects "
                    f"quantity '{target_info.quantity}' and unit "
                    f"'{target_info.unit}'."
                )
        scaler.load_state_dict(loaded.state_dict())
        scaler.sync_tensor_maps()
    else:
        if fixed_weights is None:
            fixed_weights = {}
        logging.info("Calculating scaler weights")
        trainer = Trainer(
            hypers={
                "fixed_weights": fixed_weights,
                "batch_size": batch_size,
                "per_structure_targets": list(per_structure_targets),
                **(trainer_hypers or {}),
            }
        )
        trainer._additive_models = additive_models
        trainer._is_distributed = is_distributed
        trainer.train(
            model=scaler,
            dtype=torch.float64,
            devices=[torch.device("cpu")],
            train_datasets=train_datasets,
            val_datasets=train_datasets,
            checkpoint_dir=checkpoint_dir,
        )
