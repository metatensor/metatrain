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

from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import is_auxiliary_output

from ..utils.additive import ZBL
from ..utils.dtype import dtype_to_str
from ..utils.metadata import append_metadata_references
from ..utils.sum_over_atoms import sum_over_atoms
from .modules.pet import PET as RawPET
from .utils import load_raw_pet_model, systems_to_batch_dict


class PET(torch.nn.Module):
    __supported_devices__ = ["cuda", "cpu"]
    __supported_dtypes__ = [torch.float32]
    __default_metadata__ = ModelMetadata(
        references={"architecture": ["https://arxiv.org/abs/2305.19302v3"]}
    )

    def __init__(self, model_hypers: Dict, dataset_info: DatasetInfo) -> None:
        super().__init__()
        if len(dataset_info.targets) != 1:
            raise ValueError("PET only supports a single target")
        self.target_name = next(iter(dataset_info.targets.keys()))
        target = dataset_info.targets[self.target_name]
        if not (
            target.is_scalar
            and target.quantity == "energy"
            and len(target.layout.block(0).properties) == 1
        ):
            raise ValueError(
                "PET only supports total-energy-like outputs, "
                f"but a {target.quantity} was provided"
            )
        if target.per_atom:
            raise ValueError(
                "PET only supports per-structure outputs, "
                "but a per-atom output was provided"
            )

        model_hypers["D_OUTPUT"] = 1
        model_hypers["TARGET_TYPE"] = "atomic"
        model_hypers["TARGET_AGGREGATION"] = "sum"
        for key in ["R_CUT", "CUTOFF_DELTA", "RESIDUAL_FACTOR"]:
            model_hypers[key] = float(model_hypers[key])
        self.hypers = model_hypers
        self.cutoff = float(self.hypers["R_CUT"])
        self.atomic_types: List[int] = dataset_info.atomic_types
        self.dataset_info = dataset_info
        self.pet = None
        self.is_lora_applied = False
        self.checkpoint_path: Optional[str] = None

        # last-layer feature size (for LLPR module)
        self.last_layer_feature_size = (
            self.hypers["N_GNN_LAYERS"]
            * self.hypers["HEAD_N_NEURONS"]
            * (1 + self.hypers["USE_BOND_ENERGIES"])
        )
        # if they are enabled, the edge features are concatenated
        # to the node features

        # additive models: these are handled by the trainer at training
        # time, and they are added to the output at evaluation time
        additive_models = []
        if self.hypers["USE_ZBL"]:
            additive_models.append(
                ZBL(
                    {},
                    dataset_info=DatasetInfo(
                        length_unit=dataset_info.length_unit,
                        atomic_types=self.atomic_types,
                        targets={
                            target_name: target_info
                            for target_name, target_info in dataset_info.targets.items()
                            if ZBL.is_valid_target(target_name, target_info)
                        },
                    ),
                )
            )
        self.additive_models = torch.nn.ModuleList(additive_models)

    def restart(self, dataset_info: DatasetInfo) -> "PET":
        merged_info = self.dataset_info.union(dataset_info)
        new_atomic_types = [
            at for at in merged_info.atomic_types if at not in self.atomic_types
        ]
        new_targets = {
            key: value
            for key, value in merged_info.targets.items()
            if key not in self.dataset_info.targets
        }

        if len(new_atomic_types) > 0:
            raise ValueError(
                f"New atomic types found in the dataset: {new_atomic_types}. "
                "The PET model does not support adding new atomic types."
            )

        if len(new_targets) > 0:
            raise ValueError(
                f"New targets found in the training options: {new_targets}. "
                "The PET model does not support adding new training targets."
            )

        self.dataset_info = merged_info
        self.atomic_types = sorted(self.atomic_types)
        return self

    def set_trained_model(self, trained_model: RawPET):
        self.pet = trained_model  # type: ignore

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

        structure_index = batch["batch"]
        _, counts = torch.unique(batch["batch"], return_counts=True)
        atom_index = torch.cat(
            [torch.arange(count, device=predictions.device) for count in counts]
        )
        samples_values = torch.stack([structure_index, atom_index], dim=1)
        samples = Labels(names=["system", "atom"], values=samples_values)
        empty_labels = Labels(
            names=["_"], values=torch.tensor([[0]], device=predictions.device)
        )

        # output the last-layer features for the outputs, if requested:
        if (
            f"mtt::aux::{self.target_name}_last_layer_features" in outputs
            or "features" in outputs
        ):
            ll_output_name = f"mtt::aux::{self.target_name}_last_layer_features"
            base_name = self.target_name
            if ll_output_name in outputs and base_name not in outputs:
                raise ValueError(
                    f"Features {ll_output_name} can only be requested "
                    f"if the corresponding output {base_name} is also requested."
                )
            ll_features = output["last_layer_features"]
            block = TensorBlock(
                values=ll_features,
                samples=samples,
                components=[],
                properties=Labels(
                    names=["properties"],
                    values=torch.arange(
                        ll_features.shape[1], device=predictions.device
                    ).reshape(-1, 1),
                ),
            )
            output_tmap = TensorMap(
                keys=empty_labels,
                blocks=[block],
            )
            if ll_output_name in outputs:
                ll_features_options = outputs[ll_output_name]
                if not ll_features_options.per_atom:
                    processed_output_tmap = sum_over_atoms(output_tmap)
                else:
                    processed_output_tmap = output_tmap
                output_quantities[ll_output_name] = processed_output_tmap
            if "features" in outputs:
                features_options = outputs["features"]
                if not features_options.per_atom:
                    processed_output_tmap = sum_over_atoms(output_tmap)
                else:
                    processed_output_tmap = output_tmap
                output_quantities["features"] = processed_output_tmap

        for output_name in outputs:
            if is_auxiliary_output(output_name):
                continue  # skip auxiliary outputs (not targets)
            energy_labels = Labels(
                names=["energy"], values=torch.tensor([[0]], device=predictions.device)
            )
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
                output_tmap = sum_over_atoms(output_tmap)
            output_quantities[output_name] = output_tmap

        if not self.training:
            # at evaluation, we also add the additive contributions
            for additive_model in self.additive_models:
                outputs_for_additive_model: Dict[str, ModelOutput] = {}
                for output_name, output_options in outputs.items():
                    if output_name in additive_model.outputs:
                        outputs_for_additive_model[output_name] = output_options
                additive_contributions = additive_model(
                    systems,
                    outputs_for_additive_model,
                    selected_atoms,
                )
                for output_name in additive_contributions:
                    output_quantities[output_name] = metatensor.torch.add(
                        output_quantities[output_name],
                        additive_contributions[output_name],
                    )

        return output_quantities

    @classmethod
    def load_checkpoint(cls, path: Union[str, Path]) -> "PET":
        checkpoint = torch.load(path, weights_only=False, map_location="cpu")
        hypers = checkpoint["hypers"]
        model_hypers = hypers["ARCHITECTURAL_HYPERS"]
        dataset_info = checkpoint["dataset_info"]
        model = cls(model_hypers=model_hypers, dataset_info=dataset_info)
        state_dict = checkpoint["model_state_dict"]
        dtype = next(iter(state_dict.values())).dtype
        lora_state_dict = checkpoint["lora_state_dict"]
        if lora_state_dict is not None:
            model.is_lora_applied = True
        else:
            lora_state_dict = {}
        wrapper = load_raw_pet_model(
            state_dict,
            model.hypers,
            model.atomic_types,
            checkpoint["self_contributions"],
            use_lora_peft=model.is_lora_applied,
            **lora_state_dict,
        )

        model.to(dtype).set_trained_model(wrapper)

        return model

    def export(
        self, metadata: Optional[ModelMetadata] = None
    ) -> MetatensorAtomisticModel:
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
                ),
                f"mtt::aux::{self.target_name.replace('mtt::', '')}_last_layer_features": ModelOutput(  # noqa: E501
                    unit="unitless", per_atom=True
                ),
            },
            atomic_types=self.atomic_types,
            interaction_range=interaction_range,
            length_unit=self.dataset_info.length_unit,
            supported_devices=["cpu", "cuda"],  # and not __supported_devices__
            dtype=dtype_to_str(dtype),
        )

        if metadata is None:
            metadata = ModelMetadata()

        append_metadata_references(metadata, self.__default_metadata__)

        return MetatensorAtomisticModel(self.eval(), metadata, capabilities)
