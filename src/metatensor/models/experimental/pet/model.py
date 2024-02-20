import torch
import numpy as np
from typing import Dict, List, Optional
from metatensor.torch import Labels, TensorMap, TensorBlock
from metatensor.torch.atomistic import (
    ModelCapabilities,
    ModelOutput,
    NeighborsListOptions,
    System,
)
from omegaconf import OmegaConf
from pet.molecule import batch_to_dict
from pet.pet import PET
from pet.hypers import Hypers

from ... import ARCHITECTURE_CONFIG_PATH
from .utils import systems_to_pyg_graphs


DEFAULT_HYPERS = OmegaConf.to_container(
    OmegaConf.load(ARCHITECTURE_CONFIG_PATH / "experimental.pet.yaml")
)

DEFAULT_MODEL_HYPERS = DEFAULT_HYPERS["ARCHITECTURAL_HYPERS"]

# We hardcode some of the hypers to make PET model work as a MLIP.
DEFAULT_MODEL_HYPERS.update(
    {"D_OUTPUT": 1, "TARGET_TYPE": "structural", "TARGET_AGGREGATION": "sum"}
)

ARCHITECTURE_NAME = "experimental.pet"


class Model(torch.nn.Module):
    def __init__(
        self, capabilities: ModelCapabilities, hypers: Dict = DEFAULT_MODEL_HYPERS
    ) -> None:
        super().__init__()
        self.name = ARCHITECTURE_NAME
        self.hypers = hypers
        self.cutoff = self.hypers["R_CUT"]
        self.all_species = capabilities.species
        self.capabilities = capabilities
        self.pet = PET(Hypers(self.hypers), 0.0, len(self.all_species))

    def set_trained_model(self, trained_model: torch.nn.Module) -> None:
        self.pet = trained_model

    def requested_neighbors_lists(
        self,
    ) -> List[NeighborsListOptions]:
        return [
            NeighborsListOptions(
                model_cutoff=self.cutoff,
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
        options = self.requested_neighbors_lists()[0]
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
                        device=predictions.device,
                    ),
                ),
                blocks=[
                    TensorBlock(
                        samples=Labels(
                            names=["structure"],
                            values=torch.arange(
                                len(predictions),
                                device=predictions.device,
                            ).view(-1, 1),
                        ),
                        components=[],
                        properties=Labels(
                            names=["property"],
                            values=torch.tensor(
                                len(outputs),
                                device=predictions.device,
                            ).view(1, -1),
                        ),
                        values=total_energies[output_name],
                    )
                ],
            )
        return total_energies
