from typing import Any, Dict, List, Literal, Optional

import metatensor.torch as mts
import torch
from deepmd.pt.model.model import get_standard_model
from metatensor.torch import Labels, TensorBlock, TensorMap
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
from metatrain.utils.data import TargetInfo
from metatrain.utils.data.dataset import DatasetInfo
from metatrain.utils.dtype import dtype_to_str
from metatrain.utils.metadata import merge_metadata
from metatrain.utils.scaler import Scaler
from metatrain.utils.sum_over_atoms import sum_over_atoms


# Data processing
def concatenate_structures(systems: List[System]):
    device = systems[0].positions.device
    positions = []
    species = []
    cells = []
    atom_nums: List[int] = []
    node_counter = 0

    atom_index_list: List[torch.Tensor] = []
    system_index_list: List[torch.Tensor] = []

    for i, system in enumerate(systems):
        atom_nums.append(len(system.positions))
        atom_index_list.append(torch.arange(start=0, end=len(system.positions)))
        system_index_list.append(torch.full((len(system.positions),), i))
    max_atom_num = max(atom_nums)
    atom_index = torch.cat(atom_index_list, dim=0).to(torch.int32).to(device)
    system_index = torch.cat(system_index_list, dim=0).to(torch.int32).to(device)

    positions = torch.zeros(
        (len(systems), max_atom_num, 3), dtype=systems[0].positions.dtype
    )
    species = torch.full((len(systems), max_atom_num), -1, dtype=systems[0].types.dtype)
    cells = torch.stack(
        [system.cell for system in systems]
    )  # 形状为 [batch_size, 3, 3] 或相应的晶胞形状

    for i, system in enumerate(systems):
        positions[i, : len(system.positions)] = system.positions
        species[i, : len(system.positions)] = system.types
        cells[i] = system.cell
        node_counter += len(system.positions)

    return (
        positions.to(device),
        species.to(device),
        cells.to(device),
        atom_index,
        system_index,
    )


# Model definition
class DPA3(ModelInterface):
    __checkpoint_version__ = 1
    __supported_devices__ = ["cuda", "cpu"]
    __supported_dtypes__ = [torch.float32, torch.float64]
    __default_metadata__ = ModelMetadata(
        references={
            "implementation": [
                "https://github.com/deepmodeling/deepmd-kit",
            ],
            "architecture": [
                "DPA3: https://arxiv.org/abs/2506.01686",
            ],
        }
    )

    component_labels: Dict[str, List[List[Labels]]]  # torchscript needs this

    def __init__(self, hypers: Dict, dataset_info: DatasetInfo) -> None:
        super().__init__(hypers, dataset_info, self.__default_metadata__)
        self.atomic_types = dataset_info.atomic_types
        self.dtype = self.hypers["descriptor"]["precision"]

        if self.dtype == "float64":
            self.dtype = torch.float64
        elif self.dtype == "float32":
            self.dtype = torch.float32
        else:
            raise ValueError(f"Unsupported precision: {self.dtype}")

        self.requested_nl = NeighborListOptions(
            cutoff=self.hypers["descriptor"]["repflow"]["e_rcut"],
            full_list=True,
            strict=True,
        )
        self.targets_keys = list(dataset_info.targets.keys())[0]

        self.model = get_standard_model(hypers)

        self.scaler = Scaler(hypers={}, dataset_info=dataset_info)
        self.outputs: Dict[str, ModelOutput] = {}
        self.single_label = Labels.single()

        self.num_properties: Dict[str, Dict[str, int]] = {}

        self.key_labels: Dict[str, Labels] = {}
        self.component_labels: Dict[str, List[List[Labels]]] = {}
        self.property_labels: Dict[str, List[Labels]] = {}
        for target_name, target in dataset_info.targets.items():
            self._add_output(target_name, target)

        composition_model = CompositionModel(
            hypers={},
            dataset_info=DatasetInfo(
                length_unit=dataset_info.length_unit,
                atomic_types=self.atomic_types,
                targets={
                    target_name: target_info
                    for target_name, target_info in dataset_info.targets.items()
                    if CompositionModel.is_valid_target(target_name, target_info)
                },
            ),
        )
        additive_models = [composition_model]
        self.additive_models = torch.nn.ModuleList(additive_models)

    def _add_output(self, target_name: str, target: TargetInfo) -> None:
        if not target.is_scalar:
            raise ValueError("The DPA3 architecture can only predict scalars.")
        self.num_properties[target_name] = {}
        self.key_labels[target_name] = target.layout.keys
        self.component_labels[target_name] = [
            block.components for block in target.layout.blocks()
        ]
        self.property_labels[target_name] = [
            block.properties for block in target.layout.blocks()
        ]
        self.outputs[target_name] = ModelOutput(
            quantity=target.quantity,
            unit=target.unit,
            per_atom=True,
        )

    def get_rcut(self):
        return self.model.atomic_model.get_rcut()

    def get_sel(self):
        return self.model.atomic_model.get_sel()

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        return [self.requested_nl]

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        device = systems[0].positions.device

        atype_dtype = systems[0].types.dtype

        if self.single_label.values.device != device:
            self.single_label = self.single_label.to(device)
            self.key_labels = {
                output_name: label.to(device)
                for output_name, label in self.key_labels.items()
            }
            self.component_labels = {
                output_name: [
                    [labels.to(device) for labels in components_block]
                    for components_block in components_tmap
                ]
                for output_name, components_tmap in self.component_labels.items()
            }
            self.property_labels = {
                output_name: [labels.to(device) for labels in properties_tmap]
                for output_name, properties_tmap in self.property_labels.items()
            }

        return_dict: Dict[str, TensorMap] = {}

        (positions, species, cells, atom_index, system_index) = concatenate_structures(
            systems
        )

        type_to_index = {
            atomic_type: idx for idx, atomic_type in enumerate(self.atomic_types)
        }
        type_to_index[-1] = -1

        atype = torch.tensor(
            [[type_to_index[s.item()] for s in row] for row in species],
            dtype=atype_dtype,
        ).to(positions.device)
        atype = atype.to(atype_dtype)

        if torch.all(cells == 0).item():
            box = None
        else:
            box = cells

        model_ret = self.model.forward_common(
            positions,
            atype,
            box,
            fparam=None,
            aparam=None,
            do_atomic_virial=False,
        )

        if self.model.get_fitting_net() is not None:
            model_predict = {}
            model_predict["atom_energy"] = model_ret["energy"]
            model_predict["energy"] = model_ret["energy_redu"]
            if self.model.do_grad_r("energy"):
                model_predict["force"] = model_ret["energy_derv_r"].squeeze(-2)

            else:
                model_predict["force"] = model_ret["dforce"]
            if "mask" in model_ret:
                model_predict["mask"] = model_ret["mask"]
        else:
            model_predict = model_ret
            model_predict["updated_coord"] += positions

        atomic_properties: Dict[str, TensorMap] = {}
        blocks: List[TensorBlock] = []

        system_col = system_index
        atom_col = atom_index

        values = torch.stack([system_col, atom_col], dim=0).transpose(0, 1)
        invariant_coefficients = Labels(
            names=["system", "atom"], values=values.to(device)
        )

        mask = torch.abs(model_predict["atom_energy"]) > 1e-10
        atomic_property_tensor = model_predict["atom_energy"][mask].unsqueeze(-1)

        blocks.append(
            TensorBlock(
                values=atomic_property_tensor,
                samples=invariant_coefficients,
                components=self.component_labels[self.targets_keys][0],
                properties=self.property_labels[self.targets_keys][0].to(device),
            )
        )

        atomic_properties[self.targets_keys] = TensorMap(
            self.key_labels[self.targets_keys].to(device), blocks
        )

        if selected_atoms is not None:
            for output_name, tmap in atomic_properties.items():
                atomic_properties[output_name] = mts.slice(
                    tmap, axis="samples", selection=selected_atoms
                )

        for output_name, atomic_property in atomic_properties.items():
            if outputs[output_name].per_atom:
                return_dict[output_name] = atomic_property
            else:
                # sum the atomic property to get the total property
                return_dict[output_name] = sum_over_atoms(atomic_property)

        if not self.training:
            # at evaluation, we also introduce the scaler and additive contributions
            return_dict = self.scaler(return_dict)
            for additive_model in self.additive_models:
                outputs_for_additive_model: Dict[str, ModelOutput] = {}
                for name, output in outputs.items():
                    if name in additive_model.outputs:
                        outputs_for_additive_model[name] = output
                additive_contributions = additive_model(
                    systems,
                    outputs_for_additive_model,
                    selected_atoms,
                )
                for name in additive_contributions:
                    return_dict[name] = mts.add(
                        return_dict[name],
                        additive_contributions[name],
                    )

        return return_dict

    def restart(self, dataset_info: DatasetInfo) -> "DPA3":
        # merge old and new dataset info
        merged_info = self.dataset_info.union(dataset_info)
        new_atomic_types = [
            at for at in merged_info.atomic_types if at not in self.atomic_types
        ]
        new_targets = {
            key: value
            for key, value in merged_info.targets.items()
            if key not in self.dataset_info.targets
        }
        self.has_new_targets = len(new_targets) > 0

        if len(new_atomic_types) > 0:
            raise ValueError(
                f"New atomic types found in the dataset: {new_atomic_types}. "
                "The DPA3 model does not support adding new atomic types."
            )

        # register new outputs as new last layers
        for target_name, target in new_targets.items():
            self._add_output(target_name, target)

        self.dataset_info = merged_info

        # restart the composition and scaler models
        self.additive_models[0].restart(
            dataset_info=DatasetInfo(
                length_unit=dataset_info.length_unit,
                atomic_types=self.atomic_types,
                targets={
                    target_name: target_info
                    for target_name, target_info in dataset_info.targets.items()
                    if CompositionModel.is_valid_target(target_name, target_info)
                },
            ),
        )
        self.scaler.restart(dataset_info)

        return self

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint: Dict[str, Any],
        context: Literal["restart", "finetune", "export"],
    ) -> "DPA3":
        model_data = checkpoint["model_data"]

        if context == "restart":
            model_state_dict = checkpoint["model_state_dict"]
        elif context == "finetune" or context == "export":
            model_state_dict = checkpoint["best_model_state_dict"]
            if model_state_dict is None:
                model_state_dict = checkpoint["model_state_dict"]
        else:
            raise ValueError("Unknown context tag for checkpoint loading!")

        # Create the model
        model = cls(
            hypers=model_data["model_hypers"],
            dataset_info=model_data["dataset_info"],
        )
        dtype = next(iter(model_state_dict.values())).dtype
        model.to(dtype).load_state_dict(model_state_dict)
        model.additive_models[0].sync_tensor_maps()

        # Loading the metadata from the checkpoint
        metadata = checkpoint.get("metadata", None)
        if metadata is not None:
            model.__default_metadata__ = metadata

        return model

    def export(self, metadata: Optional[ModelMetadata] = None) -> AtomisticModel:
        dtype = next(self.parameters()).dtype
        if dtype not in self.__supported_dtypes__:
            raise ValueError(f"unsupported dtype {self.dtype} for DPA3")

        # Make sure the model is all in the same dtype
        # For example, after training, the additive models could still be in
        # float64
        self.to(dtype)

        # Additionally, the composition model contains some `TensorMap`s that cannot
        # be registered correctly with Pytorch. This funciton moves them:

        self.additive_models[0].weights_to(torch.device("cpu"), torch.float64)

        interaction_ranges = [self.hypers["descriptor"]["repflow"]["e_rcut"]]
        for additive_model in self.additive_models:
            if hasattr(additive_model, "cutoff_radius"):
                interaction_ranges.append(additive_model.cutoff_radius)
        interaction_range = max(interaction_ranges)

        capabilities = ModelCapabilities(
            outputs=self.outputs,
            atomic_types=self.atomic_types,
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
    def upgrade_checkpoint(cls, checkpoint: Dict) -> Dict:
        # version is still one, there are no new versions
        return checkpoint

    def get_checkpoint(self) -> Dict:
        checkpoint = {
            "architecture_name": "experimental.dpa3",
            "model_ckpt_version": self.__checkpoint_version__,
            "metadata": self.metadata,
            "model_data": {
                "model_hypers": self.hypers,
                "dataset_info": self.dataset_info,
            },
            "model_state_dict": self.state_dict(),
            "best_model_state_dict": None,
        }
        return checkpoint

    def supported_outputs(self) -> Dict[str, ModelOutput]:
        return self.outputs
