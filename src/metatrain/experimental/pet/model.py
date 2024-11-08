import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import metatensor.torch
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.atomistic import (
    MetatensorAtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    NeighborListOptions,
    System,
)
from pet.hypers import Hypers
from pet.pet import PET as RawPET
from pet.pet import SelfContributionsWrapper

from metatrain.utils.data import DatasetInfo

from ...utils.additive import ZBL
from ...utils.dtype import dtype_to_str
from .utils import systems_to_batch_dict, update_state_dict
from .utils.fine_tuning import LoRAWrapper


logger = logging.getLogger(__name__)


class PET(torch.nn.Module):
    __supported_devices__ = ["cuda", "cpu"]
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
        self.atomic_types: List[int] = dataset_info.atomic_types
        self.dataset_info = dataset_info
        self.pet = None
        self.checkpoint_path: Optional[str] = None

        # additive models: these are handled by the trainer at training
        # time, and they are added to the output at evaluation time
        additive_models = []
        if self.hypers["USE_ZBL"]:
            additive_models.append(ZBL(model_hypers, dataset_info))
        self.additive_models = torch.nn.ModuleList(additive_models)

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
                strict=True,
            )
        ]

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        options = self.requested_neighbor_lists()[0]
        batch = systems_to_batch_dict(
            systems, options, self.atomic_types, selected_atoms
        )

        output = self.pet(batch)  # type: ignore
        predictions = output["prediction"]
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
                block = metatensor.torch.slice_block(block, "samples", selected_atoms)
            output_tmap = TensorMap(keys=empty_labels, blocks=[block])
            if not outputs[output_name].per_atom:
                output_tmap = metatensor.torch.sum_over_samples(output_tmap, "atom")
            output_quantities[output_name] = output_tmap

        if not self.training:
            # at evaluation, we also add the additive contributions
            for additive_model in self.additive_models:
                additive_contributions = additive_model(
                    systems, outputs, selected_atoms
                )
                for output_name in output_quantities:
                    if output_name.startswith("mtt::aux::"):
                        continue  # skip auxiliary outputs (not targets)
                    output_quantities[output_name] = metatensor.torch.add(
                        output_quantities[output_name],
                        additive_contributions[output_name],
                    )

        return output_quantities

    @classmethod
    def load_checkpoint(cls, path: Union[str, Path]) -> "PET":

        checkpoint = torch.load(path, weights_only=False)
        hypers = checkpoint["hypers"]
        dataset_info = checkpoint["dataset_info"]
        model = cls(
            model_hypers=hypers["ARCHITECTURAL_HYPERS"], dataset_info=dataset_info
        )

        checkpoint = torch.load(path, weights_only=False)
        state_dict = checkpoint["checkpoint"]["model_state_dict"]

        ARCHITECTURAL_HYPERS = Hypers(model.hypers)
        raw_pet = RawPET(ARCHITECTURAL_HYPERS, 0.0, len(model.atomic_types))
        if ARCHITECTURAL_HYPERS.USE_LORA_PEFT:
            lora_rank = ARCHITECTURAL_HYPERS.LORA_RANK
            lora_alpha = ARCHITECTURAL_HYPERS.LORA_ALPHA
            raw_pet = LoRAWrapper(raw_pet, lora_rank, lora_alpha)

        new_state_dict = update_state_dict(state_dict)

        dtype = next(iter(new_state_dict.values())).dtype
        raw_pet.to(dtype).load_state_dict(new_state_dict)

        self_contributions = checkpoint["self_contributions"]
        wrapper = SelfContributionsWrapper(raw_pet, self_contributions)

        model.to(dtype).set_trained_model(wrapper)

        return model

    def export(self) -> MetatensorAtomisticModel:
        dtype = next(self.parameters()).dtype
        if dtype not in self.__supported_dtypes__:
            raise ValueError(f"Unsupported dtype {self.dtype} for PET")

        # Make sure the model is all in the same dtype
        # For example, after training, the additive models could still be in
        # float64
        self.to(dtype)

        interaction_ranges = [self.hypers["N_GNN_LAYERS"] * self.cutoff]
        for additive_model in self.additive_models:
            if hasattr(additive_model, "cutoff_radius"):
                interaction_ranges.append(additive_model.cutoff_radius)
        interaction_range = max(interaction_ranges)

        capabilities = ModelCapabilities(
            outputs={
                self.target_name: ModelOutput(
                    quantity=self.dataset_info.targets[self.target_name].quantity,
                    unit=self.dataset_info.targets[self.target_name].unit,
                    per_atom=False,
                )
            },
            atomic_types=self.atomic_types,
            interaction_range=interaction_range,
            length_unit=self.dataset_info.length_unit,
            supported_devices=["cpu", "cuda"],  # and not __supported_devices__
            dtype=dtype_to_str(dtype),
        )
        return MetatensorAtomisticModel(self.eval(), ModelMetadata(), capabilities)
