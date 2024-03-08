from typing import Dict, List, Optional

import metatensor.torch
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.atomistic import (
    ModelCapabilities,
    ModelOutput,
    NeighborsListOptions,
    System,
)
from omegaconf import OmegaConf
from pet.hypers import Hypers
from pet.pet import PET

from ... import ARCHITECTURE_CONFIG_PATH
from .utils import systems_to_batch_dict


DEFAULT_HYPERS = OmegaConf.to_container(
    OmegaConf.load(ARCHITECTURE_CONFIG_PATH / "experimental.pet.yaml")
)

DEFAULT_MODEL_HYPERS = DEFAULT_HYPERS["ARCHITECTURAL_HYPERS"]

# We hardcode some of the hypers to make PET work as a MLIP.
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
        self.hypers = Hypers(hypers) if isinstance(hypers, dict) else hypers
        self.cutoff = self.hypers.R_CUT
        self.all_species: List[int] = capabilities.species
        self.capabilities = capabilities
        per_atom_output_types = [
            output.per_atom for output in self.capabilities.outputs.values()
        ]
        if any(per_atom_output_types):
            if not all(per_atom_output_types):
                raise ValueError("All outputs must be per-atom or not per-atom.")
            print(self.hypers.TARGET_TYPE)
            self.hypers.TARGET_TYPE = "atomic"
        self.pet = PET(self.hypers, 0.0, len(self.all_species))

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
        options = self.requested_neighbors_lists()[0]
        batch = systems_to_batch_dict(systems, options, self.all_species)
        predictions = self.pet(batch)
        output_quantities: Dict[str, TensorMap] = {}
        for output_name in outputs:
            empty_labels = Labels(
                names=["_"], values=torch.tensor([[0]], device=predictions.device)
            )
            if outputs[output_name].per_atom:
                structure_index = torch.repeat_interleave(
                    torch.arange(len(systems), device=predictions.device),
                    torch.tensor(
                        [len(system) for system in systems], device=predictions.device
                    ),
                )
                atom_index = torch.cat(
                    [
                        torch.arange(len(system), device=predictions.device)
                        for system in systems
                    ]
                )
                samples_values = torch.stack([structure_index, atom_index], dim=1)
                samples = Labels(names=["system", "atom"], values=samples_values)
                block = TensorBlock(
                    samples=samples,
                    components=[],
                    properties=empty_labels,
                    values=predictions,
                )
            else:
                samples_values = torch.arange(
                    len(systems), device=predictions.device
                ).view(-1, 1)
                samples = Labels(names=["system"], values=samples_values)
                block = TensorBlock(
                    samples=samples,
                    components=[],
                    properties=empty_labels,
                    values=predictions,
                )
            if selected_atoms is not None:
                block = metatensor.torch.slice_block(block, "samples", selected_atoms)
            output_quantities[output_name] = TensorMap(
                keys=empty_labels, blocks=[block]
            )
        return output_quantities
