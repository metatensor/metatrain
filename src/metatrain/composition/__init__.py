import logging
from typing import List, Union

import torch
from torch import nn

from metatrain.utils.data import Dataset
from metatrain.utils.data.dataset import Subset

from ._base_composition import FixedCompositionWeights
from .model import CompositionModel
from .trainer import Trainer


__model__ = CompositionModel
__trainer__ = Trainer

__authors__ = [
    ("Paolo Pegolo <paolo.pegolo@epfl.ch>", "@ppegolo"),
]

__maintainers__ = [
    ("Paolo Pegolo <paolo.pegolo@epfl.ch>", "@ppegolo"),
]


def train_or_load_composition_model(
    composition_model: CompositionModel,
    atomic_baseline: FixedCompositionWeights | str,
    train_datasets: List[Union[Dataset, Subset]],
    other_additive_models: List[nn.Module],
    batch_size: int,
    is_distributed: bool,
    checkpoint_dir: str = "",
) -> None:
    """
    Train the composition model from data or load pre-trained weights.

    This is the single source of truth for how to set up a composition model
    for use as an additive baseline by any architecture.

    :param composition_model: The composition model to train or load into
    :param atomic_baseline: Fixed weights dict, or path to a checkpoint
    :param train_datasets: Training datasets
    :param other_additive_models: Other additive models (e.g. ZBL) to
        subtract before fitting
    :param batch_size: Batch size for data loading
    :param is_distributed: Whether training is distributed
    :param checkpoint_dir: Directory to save the composition model checkpoint
    """
    if isinstance(atomic_baseline, str):
        logging.info(f"Loading composition model from {atomic_baseline}")
        checkpoint = torch.load(atomic_baseline, map_location="cpu", weights_only=False)
        checkpoint = CompositionModel.upgrade_checkpoint(checkpoint)
        loaded = CompositionModel.load_checkpoint(checkpoint, context="export")
        loaded.sync_tensor_maps()
        composition_model.load_state_dict(loaded.state_dict())
        composition_model.sync_tensor_maps()
    else:
        assert isinstance(atomic_baseline, dict)
        logging.info("Calculating composition weights")
        trainer = Trainer(hypers={"atomic_baseline": atomic_baseline})
        trainer._additive_models = other_additive_models
        trainer._is_distributed = is_distributed
        trainer.train(
            model=composition_model,
            dtype=torch.float64,
            devices=[torch.device("cpu")],
            train_datasets=train_datasets,
            val_datasets=train_datasets,
            checkpoint_dir=checkpoint_dir,
        )
