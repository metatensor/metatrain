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
    {"D_OUTPUT": 1, "TARGET_TYPE": "atomic", "TARGET_AGGREGATION": "sum"}
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
        self.all_species: List[int] = capabilities.atomic_types
        self.capabilities = capabilities
        self.pet = PET(self.hypers, 0.0, len(self.all_species))

    def set_trained_model(self, trained_model: torch.nn.Module) -> None:
        self.pet = trained_model

    def requested_neighbors_lists(
        self,
    ) -> List[NeighborsListOptions]:
        return [
            NeighborsListOptions(
                cutoff=self.cutoff,
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
            energy_labels = Labels(
                names=["energy"], values=torch.tensor([[0]], device=predictions.device)
            )
            empty_labels = Labels(
                names=["_"], values=torch.tensor([[0]], device=predictions.device)
            )
            structure_index = batch["batch"]
            _, counts = torch.unique(batch["batch"], return_counts=True)
            atom_index = torch.cat(
                [torch.arange(count, device=predictions.device) for count in counts]
            )
            samples_values = torch.stack([structure_index, atom_index], dim=1)
            samples = Labels(names=["system", "atom"], values=samples_values)
            block = TensorBlock(
                samples=samples,
                components=[],
                properties=energy_labels,
                values=predictions,
            )
            if selected_atoms is not None:
                block = metatensor.torch.slice_block(
                    block, axis="samples", labels=selected_atoms
                )
            output_tmap = TensorMap(keys=empty_labels, blocks=[block])
            if not outputs[output_name].per_atom:
                output_tmap = metatensor.torch.sum_over_samples(output_tmap, "atom")
            output_quantities[output_name] = output_tmap
        return output_quantities
