from typing import Any, Dict, List, Literal, Optional

import mace.modules as mace_modules
from mace.modules import MACE

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.operations._add import _add_block_block
from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    NeighborListOptions,
    System,
)

from metatrain.utils.abc import ModelInterface
from metatrain.utils.additive import CompositionModel
from metatrain.utils.data import DatasetInfo, TargetInfo
from metatrain.utils.dtype import dtype_to_str
from metatrain.utils.metadata import merge_metadata
from metatrain.utils.scaler import Scaler
from metatrain.utils.sum_over_atoms import sum_over_atoms

from e3nn import o3

from .utils.structures import create_batch

def add_contribution(
    values: Dict[str, TensorMap],
    systems: List[System],
    outputs: Dict[str, ModelOutput],
    additive_model: CompositionModel,
    selected_atoms: Optional[Labels] = None,
) -> None:
    
    outputs_for_additive_model: Dict[str, ModelOutput] = {}
    for name, output in outputs.items():
        if name in additive_model.outputs:
            outputs_for_additive_model[name] = output
    additive_contributions = additive_model.forward(
        systems,
        outputs_for_additive_model,
        selected_atoms,
    )
    for name in additive_contributions:
        # # TODO: uncomment this after metatensor.torch.add is updated to
        # # handle sparse sums
        # return_dict[name] = metatensor.torch.add(
        #     return_dict[name],
        #     additive_contributions[name].to(
        #         device=return_dict[name].device,
        #         dtype=return_dict[name].dtype
        #         ),
        # )

        # TODO: "manual" sparse sum: update to metatensor.torch.add after
        # sparse sum is implemented in metatensor.operations
        output_blocks: List[TensorBlock] = []
        for k, b in values[name].items():
            if k in additive_contributions[name].keys:
                output_blocks.append(
                    _add_block_block(
                        b,
                        additive_contributions[name]
                        .block(k)
                        .to(device=b.device, dtype=b.dtype),
                    )
                )
            else:
                output_blocks.append(b)
        values[name] = TensorMap(values[name].keys, output_blocks)

def e3nn_to_tensormap(
    target_values: torch.Tensor,
    sample_labels: Labels,
    target_info: TargetInfo,
    output_name: str,
    outputs: Dict[str, ModelOutput],
) -> TensorMap:
    
    blocks: list[TensorBlock] = []
    pointer = 0
    for i in range(len(target_info.component_labels)):

        components = target_info.component_labels[i]
        properties = target_info.property_labels[i]

        has_components = len(components) > 0
        n_components = len(components[0]) if has_components else 1
        n_properties = len(properties)

        end = pointer + n_components * n_properties

        values = target_values[:, pointer:end].reshape(
            -1, n_properties, n_components,
        ).transpose(1, 2)

        if target_info.is_cartesian and n_components == 3:
            # Go back from YZX to XYZ
            values = values[:, [2, 0, 1], :]

        if not has_components:
            # Remove the components dimension if there are no components
            values = values.squeeze(1)

        blocks.append(
            TensorBlock(
                values=values,
                samples=sample_labels,
                components=components,
                properties=properties,
            )
        )
        pointer = end

    atom_target = TensorMap(
        keys=target_info.layout.keys,
        blocks=blocks
    )

    return sum_over_atoms(atom_target) if not outputs[output_name].per_atom else atom_target

def get_system_indices_and_labels(
    systems: List[System], device: torch.device
) -> tuple[torch.Tensor, Labels]:
    
    system_indices = torch.concatenate(
    [
        torch.full(
                (len(system),),
                i_system,
                device=device,
            )
            for i_system, system in enumerate(systems)
        ],
    )

    sample_values = torch.stack(
        [
            system_indices,
            torch.concatenate(
                [
                    torch.arange(
                        len(system),
                        device=device,
                    )
                    for system in systems
                ],
            ),
        ],
        dim=1,
    )
    sample_labels = Labels(
        names=["system", "atom"],
        values=sample_values,
    )
    return system_indices, sample_labels

class MetaMACE(ModelInterface):
    """Interface of MACE for metatrain."""

    __checkpoint_version__ = 1
    __supported_devices__ = ["cuda", "cpu"]
    __supported_dtypes__ = [torch.float64, torch.float32]
    __default_metadata__ = ModelMetadata(
        # references={"architecture": ["https://arxiv.org/abs/2305.19302v3"]}
    )

    dataset_info: DatasetInfo

    def __init__(self, model_hypers: Dict, dataset_info: DatasetInfo) -> None:
        super().__init__(model_hypers, dataset_info, self.__default_metadata__)
 
        self.register_buffer(
            "atomic_types_to_species_index", torch.zeros(max(dataset_info.atomic_types) + 1, dtype=torch.int64)
        )
        for i, atomic_type in enumerate(dataset_info.atomic_types):
            self.atomic_types_to_species_index[atomic_type] = i

        self.requested_nl = NeighborListOptions(
            cutoff=self.hypers["cutoff"],
            full_list=True,
            strict=True,
        )

        self.cutoff = float(self.hypers["cutoff"])

        self.mace_model = MACE(
            r_max=self.cutoff,
            num_bessel=model_hypers["num_radial_basis"],
            num_polynomial_cutoff=model_hypers["num_cutoff_basis"],
            max_ell=model_hypers["max_ell"],
            interaction_cls=mace_modules.interaction_classes[model_hypers["interaction"]],
            num_interactions=model_hypers["num_interactions"],
            num_elements=len(dataset_info.atomic_types),
            hidden_irreps=o3.Irreps(model_hypers["hidden_irreps"]),
            edge_irreps=o3.Irreps(model_hypers["edge_irreps"]) if "edge_irreps" in model_hypers["edge_irreps"] else None,
            atomic_energies=torch.zeros(len(dataset_info.atomic_types)),
            apply_cutoff=model_hypers["apply_cutoff"],
            avg_num_neighbors=model_hypers["avg_num_neighbors"],
            atomic_numbers=torch.arange(len(dataset_info.atomic_types)),
            pair_repulsion=model_hypers["pair_repulsion"],
            distance_transform=model_hypers["distance_transform"],
            correlation=model_hypers["correlation"],
            gate=mace_modules.gate_dict[model_hypers["gate"]] if model_hypers["gate"] is not None else None,
            interaction_cls_first=mace_modules.interaction_classes[model_hypers["interaction_first"]],
            MLP_irreps=o3.Irreps(model_hypers["MLP_irreps"]),
            radial_MLP=model_hypers["radial_MLP"],
            radial_type=model_hypers["radial_type"],
            use_embedding_readout=model_hypers["use_embedding_readout"],
            use_last_readout_only=model_hypers["use_last_readout_only"],
            use_agnostic_product=model_hypers["use_agnostic_product"],
        )

        self.outputs = {
            "features": ModelOutput(unit="", per_atom=True)
        }
        self.heads: Dict[str, torch.nn.Module] = torch.nn.ModuleDict()
        for target_name, target_info in dataset_info.targets.items():
            self._add_output(target_name, target_info)

        composition_model = CompositionModel(
            hypers={},
            dataset_info=DatasetInfo(
                length_unit=dataset_info.length_unit,
                atomic_types=self.dataset_info.atomic_types,
                targets={
                    target_name: target_info
                    for target_name, target_info in dataset_info.targets.items()
                    if CompositionModel.is_valid_target(target_name, target_info)
                },
            ),
        )

        additive_models = [composition_model]
        self.additive_models = torch.nn.ModuleList(additive_models)

        self.scaler = Scaler(hypers={}, dataset_info=dataset_info)
    
    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        
        if selected_atoms is not None:
            raise NotImplementedError(
                "selected_atoms is not supported in MetaMACE for now. "
            )
        
        # Move everything to the same device
        device = systems[0].device
        self.dataset_info = self.dataset_info.to(device=device)

        # Create the batch to pass as input for MACE.
        # THIS PROBABLY SHOULD BE MOVED OUTSIDE THE MODEL!!
        # (But I don't know if this would affect the interfaces e.g. with
        # ASE, LAMMPS, etc.)
        data = create_batch(
            systems=systems,
            neighbor_list_options=self.requested_nl,
            atomic_types_to_species_index=self.atomic_types_to_species_index,
            n_types=len(self.dataset_info.atomic_types),
            device=device,
        )

        # Change coordinates to YZX
        data["positions"] = data["positions"][:, [1, 2, 0]]

        # Run MACE and extract the node features.
        mace_output = self.mace_model(data, training=self.training)
        node_features = mace_output["node_feats"]
        assert node_features is not None # For torchscript

        # Get the labels for the samples (system and atom of each value)
        _, sample_labels = get_system_indices_and_labels(
            systems, device
        )

        # Run all heads and collect outputs as TensorMaps
        return_dict: Dict[str, TensorMap] = {}
        for output_name, head in self.heads.items():
            node_target = head.forward(node_features, weight=None, bias=None)
            target_info = self.dataset_info.targets[output_name]

            return_dict[output_name] = e3nn_to_tensormap(
                node_target,
                sample_labels,
                target_info,
                output_name,
                outputs, 
            )

        if not self.training:
            # at evaluation, we also introduce the scaler and additive contributions
            return_dict = self.scaler(systems,return_dict)
            for additive_model in self.additive_models:
                add_contribution(
                    return_dict,
                    systems,
                    outputs,
                    additive_model,
                    selected_atoms
                )
                

        return return_dict

    def _add_output(self, target_name: str, target_info: TargetInfo) -> None:
        # We don't support Cartesian tensors with rank > 1
        if target_info.is_cartesian:
            if len(target_info.layout.block().components) > 1:
                raise ValueError(
                    "MetaMACE does not support Cartesian tensors with rank > 1."
                )

        # Get the multiplicity and irrep for each target block
        irreps = []
        for key, block in target_info.layout.items():
            multiplicity = len(block.properties.values)

            if target_info.is_scalar:
                irreps.append((multiplicity, (0, 1)))
            elif target_info.is_spherical:
                l = int(key["o3_lambda"])
                irreps.append((multiplicity, (l, (-1)**l)))                  
            elif target_info.is_cartesian:
                l = 1
                irreps.append((multiplicity, (l, (-1)**l)))

        self.outputs[target_name] = ModelOutput(
            quantity=target_info.quantity,
            unit=target_info.unit,
            per_atom=True,
        )

        hidden_irreps = o3.Irreps(self.hypers["hidden_irreps"])
        n_scalars = hidden_irreps.count((0, 1)) 
        mace_out_irreps = hidden_irreps * (self.hypers["num_interactions"] - 1) + o3.Irreps([(n_scalars, (0, 1))])

        self.heads[target_name] = o3.Linear(
            irreps_in=mace_out_irreps,
            irreps_out=o3.Irreps(irreps)
        )

        self.heads[target_name].to(torch.float64)

        ll_features_name = (
            f"mtt::aux::{target_name.replace('mtt::', '')}_last_layer_features"
        )
        self.outputs[ll_features_name] = ModelOutput(per_atom=True)

    def supported_outputs(self) -> Dict[str, ModelOutput]:
        return self.outputs
    
    def requested_neighbor_lists(
        self,
    ) -> List[NeighborListOptions]:
        return [self.requested_nl]

    def restart(self, dataset_info: DatasetInfo) -> "MetaMACE":
        # Check that the new dataset info does not contain new atomic types
        if new_atomic_types := set(dataset_info.atomic_types) - set(self.dataset_info.atomic_types):
            raise ValueError(
                f"New atomic types found in the dataset: {new_atomic_types}. "
                "The MACE model does not support adding new atomic types."
            )

        # Merge the old dataset info with the new one
        merged_info = self.dataset_info.union(dataset_info)

        # Check if there are new targets
        new_targets = {
            key: value
            for key, value in merged_info.targets.items()
            if key not in self.dataset_info.targets
        }
        self.has_new_targets = len(new_targets) > 0

        # Add extra heads for the new targets
        for target_name, target in new_targets.items():
            self._add_output(target_name, target)

        self.dataset_info = merged_info

        # Restart the composition and scaler models
        self.additive_models[0].restart(
            dataset_info=DatasetInfo(
                length_unit=dataset_info.length_unit,
                atomic_types=self.dataset_info.atomic_types,
                targets={
                    target_name: target_info
                    for target_name, target_info in dataset_info.targets.items()
                    if CompositionModel.is_valid_target(target_name, target_info)
                },
            ),
        )
        self.scaler.restart(dataset_info)

        return self
    
    def export(self, metadata: Optional[ModelMetadata] = None) -> AtomisticModel:
        dtype = next(self.parameters()).dtype
        if dtype not in self.__supported_dtypes__:
            raise ValueError(f"unsupported dtype {dtype} for MACE")

        # Make sure the model is all in the same dtype
        # For example, after training, the additive models could still be in
        # float64
        self.to(dtype)

        # Additionally, the composition model contains some `TensorMap`s that cannot
        # be registered correctly with Pytorch. This function moves them:
        self.additive_models[0].weights_to(torch.device("cpu"), torch.float64)

        interaction_ranges = [self.hypers["num_interactions"] * self.hypers["cutoff"]]
        interaction_range = max(interaction_ranges)

        capabilities = ModelCapabilities(
            outputs=self.outputs,
            atomic_types=self.dataset_info.atomic_types,
            interaction_range=interaction_range,
            length_unit=self.dataset_info.length_unit,
            supported_devices=self.__supported_devices__,
            dtype=dtype_to_str(dtype),
        )

        if metadata is None:
            metadata = self.__default_metadata__
        else:
            metadata = merge_metadata(self.__default_metadata__, metadata)

        return AtomisticModel(self.eval(), metadata, capabilities)
    
    @classmethod
    def load_checkpoint(
        cls,
        checkpoint: Dict[str, Any],
        context: Literal["restart", "export"],
    ) -> "MetaMACE":
        model_data = checkpoint["model_data"]

        if context == "restart":
            model_state_dict = checkpoint["model_state_dict"]
        elif context == "export":
            model_state_dict = checkpoint["best_model_state_dict"]
            if model_state_dict is None:
                model_state_dict = checkpoint["model_state_dict"]
        else:
            raise ValueError("Unknown context tag for checkpoint loading!")

        # Create the model
        model = cls(**model_data)
        dtype = None
        for k, v in model_state_dict.items():
            if k.endswith(".weight"):
                dtype = v.dtype
                break
        else:
            raise ValueError("Couldn't infer dtype from the checkpoint file")
        model.to(dtype).load_state_dict(model_state_dict)
        model.additive_models[0].sync_tensor_maps()

        model.metadata = merge_metadata(model.metadata, checkpoint.get("metadata"))

        return model
    
    @classmethod
    def upgrade_checkpoint(cls, checkpoint: Dict) -> Dict:
        if checkpoint["model_ckpt_version"] != cls.__checkpoint_version__:
            raise RuntimeError(
                f"Unable to upgrade the checkpoint: the checkpoint is using model "
                f"version {checkpoint['model_ckpt_version']}, while the current model "
                f"version is {cls.__checkpoint_version__}."
            )
        
        return checkpoint

    def get_checkpoint(self) -> Dict:
        model_state_dict = self.state_dict()
        checkpoint = {
            "architecture_name": "experimental.mace",
            "model_ckpt_version": self.__checkpoint_version__,
            "metadata": self.metadata,
            "model_data": {
                "model_hypers": self.hypers,
                "dataset_info": self.dataset_info.to(device="cpu"),
            },
            "model_state_dict": model_state_dict,
            "best_model_state_dict": None,
        }
        return checkpoint
