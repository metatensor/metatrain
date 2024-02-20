import torch

from typing import Dict, List, Optional
from metatensor.torch import Labels, TensorMap
from metatensor.torch.atomistic import (
    ModelCapabilities,
    ModelOutput,
    NeighborsListOptions,
    System,
)
from omegaconf import OmegaConf
from pet.molecule import batch_to_dict
from pet.pet import PET

from ... import ARCHITECTURE_CONFIG_PATH
from .utils import systems_to_pyg_graphs


DEFAULT_HYPERS = OmegaConf.to_container(
    OmegaConf.load(ARCHITECTURE_CONFIG_PATH / "experimental.pet.yaml")
)

DEFAULT_MODEL_HYPERS = DEFAULT_HYPERS["ARCHITECTURAL_HYPERS"]

ARCHITECTURE_NAME = "experimental.pet"


class Model(torch.nn.Module):
    def __init__(
        self, capabilities: ModelCapabilities, hypers: Dict = DEFAULT_MODEL_HYPERS
    ) -> None:
        super().__init__()
        self.name = ARCHITECTURE_NAME
        self.hypers = hypers
        self.all_species = capabilities.species
        self.pet = PET(hypers, 0.0, len(self.all_species))

    def requested_neighbors_lists(
        self,
    ) -> List[NeighborsListOptions]:
        return [
            NeighborsListOptions(
                model_cutoff=self.hypers.R_CUT,
                full_list=True,
            )
        ]

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        if selected_atoms is not None:
            raise NotImplementedError("PET does not support selected atoms.")
        options = self.requested_neighbors_lists[0]
        batch = systems_to_pyg_graphs(systems, options, self.all_species)
        predictions = self.pet(batch_to_dict(batch))
        total_energies: Dict[str, TensorMap] = {}
        for output_name in outputs:
            total_energies[output_name] = predictions
            total_energies[output_name] = TensorMap(
                keys=Labels(
                    names=["lambda", "sigma"],
                    values=torch.tensor(
                        [[0, 1]],
                        device=total_energies[output_name].block(0).values.device,
                    ),
                ),
                blocks=[total_energies[output_name].block()],
            )

        return total_energies
