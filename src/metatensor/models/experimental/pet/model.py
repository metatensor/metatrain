import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import metatensor.torch
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.atomistic import (
    MetatensorAtomisticModel,
    ModelCapabilities,
    ModelOutput,
    NeighborListOptions,
    System,
)
from pet.pet import PET as RawPET

from metatensor.models.utils.data import DatasetInfo

from ...utils.dtype import dtype_to_str
from ...utils.export import export
from ...utils.io import check_suffix
from .utils import systems_to_batch_dict


logger = logging.getLogger(__name__)


class PET(torch.nn.Module):
    __supported_devices__ = ["cuda"]
    __supported_dtypes__ = [torch.float32]

    def __init__(self, model_hypers: Dict, dataset_info: DatasetInfo) -> None:
        super().__init__()
        if len(dataset_info.targets) != 1:
            raise ValueError("PET only supports a single target")
        self.target_name = next(iter(dataset_info.targets.keys()))
        if dataset_info.targets[self.target_name].quantity != "energy":
            raise ValueError("PET only supports energies as target")

        model_hypers["D_OUTPUT"] = 1
        model_hypers["TARGET_TYPE"] = "atomic"
        model_hypers["TARGET_AGGREGATION"] = "sum"
        self.hypers = model_hypers
        self.cutoff = self.hypers["R_CUT"]
        self.species: List[int] = dataset_info.atomic_types
        self.dataset_info = dataset_info
        self.pet = None
        self.checkpoint_path: Optional[str] = None

    def restart(self, dataset_info: DatasetInfo) -> "PET":
        if dataset_info != self.dataset_info:
            raise ValueError(
                "PET cannot be restarted with different dataset information"
            )
        return self

    def set_trained_model(self, trained_model: RawPET) -> None:
        self.pet = trained_model

    def requested_neighbor_lists(
        self,
    ) -> List[NeighborListOptions]:
        return [
            NeighborListOptions(
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
        options = self.requested_neighbor_lists()[0]
        batch = systems_to_batch_dict(systems, options, self.species, selected_atoms)

        predictions = self.pet(batch)  # type: ignore
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

    def save_checkpoint(self, path: Union[str, Path]):
        torch.save(
            {
                "model_hypers": {
                    "model_hypers": self.hypers,
                    "dataset_info": self.dataset_info,
                },
                "model_state_dict": self.state_dict(),
            },
            check_suffix(path, ".ckpt"),
        )

    @classmethod
    def load_checkpoint(cls, path: Union[str, Path]) -> "PET":

        # Load the model and the metadata
        model_dict = torch.load(path)

        # Create the model
        model = cls(**model_dict["model_hypers"])

        # Load the model weights
        model.load_state_dict(model_dict["model_state_dict"])

        return model

    def export(self) -> MetatensorAtomisticModel:
        dtype = next(self.parameters()).dtype
        if dtype not in self.__supported_dtypes__:
            raise ValueError(f"Unsupported dtype {self.dtype} for PET")

        capabilities = ModelCapabilities(
            outputs={
                self.target_name: ModelOutput(
                    quantity=self.dataset_info.targets[self.target_name].quantity,
                    unit=self.dataset_info.targets[self.target_name].unit,
                    per_atom=False,
                )
            },
            atomic_types=self.species,
            interaction_range=self.cutoff,
            length_unit=self.dataset_info.length_unit,
            supported_devices=["cpu", "cuda"],  # and not __supported_devices__
            dtype=dtype_to_str(dtype),
        )
        return export(model=self, model_capabilities=capabilities)
