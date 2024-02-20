import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import rascaline
import torch
from metatensor.learn.data import DataLoader
from metatensor.learn.data.dataset import _BaseDataset
from metatensor.torch.atomistic import ModelCapabilities, NeighborsListOptions, System

from ...utils.composition import calculate_composition_weights
from ...utils.compute_loss import compute_model_loss
from ...utils.data import (
    check_datasets,
    collate_fn,
    combine_dataloaders,
    get_all_targets,
)
from ...utils.extract_targets import get_outputs_dict
from ...utils.info import finalize_aggregated_info, update_aggregated_info
from ...utils.neighbors_lists import get_system_with_neighbors_lists
from ...utils.logging import MetricLogger
from ...utils.loss import TensorMapDictLoss
from ...utils.merge_capabilities import merge_capabilities
from ...utils.model_io import load_checkpoint, save_model
from .utils import systems_to_pyg_graphs
from .model import DEFAULT_HYPERS, Model


logger = logging.getLogger(__name__)

# disable rascaline logger
rascaline.set_logging_callback(lambda x, y: None)

# Filter out the second derivative and device warnings from rascaline-torch
warnings.filterwarnings("ignore", category=UserWarning, message="second derivative")
warnings.filterwarnings(
    "ignore", category=UserWarning, message="Systems data is on device"
)


def train(
    train_datasets: List[Union[_BaseDataset, torch.utils.data.Subset]],
    validation_datasets: List[Union[_BaseDataset, torch.utils.data.Subset]],
    requested_capabilities: ModelCapabilities,
    hypers: Dict = DEFAULT_HYPERS,
    continue_from: Optional[str] = None,
    output_dir: str = ".",
    device_str: str = "cpu",
):

    if len(train_datasets) != 1:
        raise ValueError("PET only supports a single training dataset")
    if len(validation_datasets) != 1:
        raise ValueError("PET only supports a single validation dataset")
    
    train_dataset = train_datasets[0]
    validation_dataset = validation_datasets[0]

    # dummy dataloaders due to https://github.com/lab-cosmo/metatensor/issues/521
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
    )
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # only energies or energies and forces?
    do_forces = next(iter(next(iter(train_dataset))[1].values())).values.has_gradient("positions")
    target_name = ...

    # TODO: 5.0 is hardcoded, should be a hyperparameter
    requested_nl = NeighborsListOptions(cutoff=5.0, full_list=True)
    all_species = requested_capabilities.species

    pyg_train_dataset = []
    for (system,), targets in train_dataloader:
        system = get_system_with_neighbors_lists(system, get_system_with_neighbors_lists(system, [requested_nl]))
        pyg_graph = systems_to_pyg_graphs([system], requested_nl, all_species)
        pyg_graph.update({'energy': targets[target_name].block().values.squeeze(-1)})
        if do_forces:
            pyg_graph.update({'forces': targets[target_name].block().gradient('positions').values.squeeze(-1)})
    pyg_train_dataset = torch.utils.data.Dataset(pyg_train_dataset)

    pyg_validation_dataset = []
    for (system,), _ in validation_dataloader:
        system = get_system_with_neighbors_lists(system, get_system_with_neighbors_lists(system, [requested_nl]))
        pyg_graph = systems_to_pyg_graphs([system], requested_nl, all_species)
        pyg_graph.update({'energy': targets[target_name].block().values.squeeze(-1)})
        if do_forces:
            pyg_graph.update({'forces': targets[target_name].block().gradient('positions').values.squeeze(-1)})
    pyg_validation_dataset = torch.utils.data.Dataset(pyg_validation_dataset)

