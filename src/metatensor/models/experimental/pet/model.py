from typing import Dict, List, Optional

import metatensor.torch
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.atomistic import ModelCapabilities, ModelOutput, System
from omegaconf import OmegaConf

import torch
import numpy as np
from torch_geometric.data import Data, Batch

from metatensor.torch.atomistic import NeighborsListOptions
from pet.molecule import NeighborIndexConstructor, batch_to_dict
from .utils.systems_to_pyg_graphs import systems_to_pyg_graphs

from ... import ARCHITECTURE_CONFIG_PATH
from ...utils.composition import apply_composition_contribution


ARCHITECTURE_NAME = "experimental.soap_bpnn"
DEFAULT_HYPERS = OmegaConf.to_container(
    OmegaConf.load(ARCHITECTURE_CONFIG_PATH / f"{ARCHITECTURE_NAME}.yaml")
)
DEFAULT_MODEL_HYPERS = DEFAULT_HYPERS["model"]


class PETMetatensorWrapper(torch.nn.Module):
    def __init__(self, pet_model, all_species):
        super(PETMetatensorWrapper, self).__init__()
        self.pet_model = pet_model
        self.all_species = all_species
        
    def forward(self, systems):
        options = NeighborsListOptions(model_cutoff=self.pet_model.hypers.R_CUT,
                                       full_list=True)
        batch = systems_to_pyg_graphs(systems, options, self.all_species)
        #print(batch_to_dict(batch))
        return self.pet_model(batch_to_dict(batch))
