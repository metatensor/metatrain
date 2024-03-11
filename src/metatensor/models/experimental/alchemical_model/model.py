from typing import Dict, List, Optional

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.atomistic import (
    ModelCapabilities,
    ModelOutput,
    NeighborsListOptions,
    System,
)
from omegaconf import OmegaConf
from torch_alchemical.models import AlchemicalModel

from ... import ARCHITECTURE_CONFIG_PATH
from .utils import systems_to_torch_alchemical_batch


DEFAULT_HYPERS = OmegaConf.to_container(
    OmegaConf.load(ARCHITECTURE_CONFIG_PATH / "experimental.alchemical_model.yaml")
)

DEFAULT_MODEL_HYPERS = DEFAULT_HYPERS["model"]

ARCHITECTURE_NAME = "experimental.alchemical_model"


class Model(torch.nn.Module):
    def __init__(
        self, capabilities: ModelCapabilities, hypers: Dict = DEFAULT_MODEL_HYPERS
    ) -> None:
        super().__init__()
        self.name = ARCHITECTURE_NAME
        self.hypers = hypers
        self.cutoff = self.hypers["soap"]["cutoff"]
        self.all_species: List[int] = capabilities.atomic_types
        self.capabilities = capabilities
        self.alchemical_model = AlchemicalModel(
            unique_numbers=self.all_species, **hypers["soap"], **hypers["bpnn"]
        )

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
        if selected_atoms is not None:
            raise NotImplementedError(
                "Alchemical Model does not support selected atoms."
            )
        options = self.requested_neighbors_lists()[0]
        batch = systems_to_torch_alchemical_batch(systems, options)
        predictions = self.alchemical_model(
            positions=batch["positions"],
            cells=batch["cells"],
            numbers=batch["numbers"],
            edge_indices=batch["edge_indices"],
            edge_offsets=batch["edge_offsets"],
            batch=batch["batch"],
        )

        total_energies: Dict[str, TensorMap] = {}
        for output_name in outputs:
            empty_label = Labels(
                "_", torch.zeros((1, 1), dtype=torch.int32, device=predictions.device)
            )
            samples = Labels(
                names=["system"],
                values=torch.arange(
                    len(predictions),
                    device=predictions.device,
                ).view(-1, 1),
            )
            block = TensorBlock(
                samples=samples,
                components=[],
                properties=empty_label,
                values=predictions,
            )
            total_energies[output_name] = TensorMap(
                keys=empty_label,
                blocks=[block],
            )
        return total_energies

    def set_composition_weights(
        self,
        input_composition_weights: torch.Tensor,
        species: List[int],
    ) -> None:
        """Set the composition weights for a given output."""
        input_composition_weights = input_composition_weights.to(
            dtype=self.alchemical_model.composition_weights.dtype,
            device=self.alchemical_model.composition_weights.device,
        )
        index = [self.all_species.index(s) for s in species]
        composition_weights = input_composition_weights[:, index]
        self.alchemical_model.set_composition_weights(composition_weights)

    def set_normalization_factor(self, normalization_factor: torch.Tensor) -> None:
        """Set the normalization factor for output of the model."""
        self.alchemical_model.set_normalization_factor(normalization_factor)

    def set_basis_normalization_factor(
        self, basis_normalization_factor: torch.Tensor
    ) -> None:
        """Set the normalization factor for the basis functions of the model."""
        self.alchemical_model.set_basis_normalization_factor(basis_normalization_factor)

    def set_energies_scale_factor(self, energies_scale_factor: torch.Tensor) -> None:
        """Set the energies scale factor for the model."""
        self.alchemical_model.set_energies_scale_factor(energies_scale_factor)
