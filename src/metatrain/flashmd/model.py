from math import prod
from pathlib import Path
from typing import Dict, List, Optional, Union

import metatensor.torch
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    NeighborListOptions,
    System,
)

from metatrain.utils.data import DatasetInfo, TargetInfo
from metatrain.utils.scaler import Scaler
from metatrain.flashmd.modules.encoder import Encoder
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.pet.modules.structures import systems_to_batch
from metatrain.pet import PET
from metatrain.utils.data.target_info import is_auxiliary_output

class FlashMDPET(torch.nn.Module):
    def __init__(self, model_hypers: dict, dataset_info: DatasetInfo) -> None:
        super().__init__()
        
        # NOTE: Here, we ignore the whole only-one-target discussion in PET because that should be part of the trainer/loss function for FlashMD.

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

        # TODO: Decide if we want additive models as in PET.
        self.additive_models = torch.nn.ModuleList([])

    # NOTE: According to PET-style, the model comes from outside, and can be set as a raw model.
    def set_trained_model(self, trained_model: PET):
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

    def forward(self, systems: List[System], outputs: Dict[str, ModelOutput], selected_atoms: Optional[Labels] = None) -> Dict[str, TensorMap]:
        # TODO: Here, we're assuming to be dealing with a direct model. Eventually, we might want to add support for an action-based model as well.
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
        for target_name in ["mtt::delta_q", "mtt::delta_p"]:
            if (
                f"mtt::aux::{target_name}_last_layer_features" in outputs
                or "features" in outputs
            ):
                ll_output_name = f"mtt::aux::{target_name}_last_layer_features"
                base_name = target_name
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
                        processed_output_tmap = metatensor.torch.sum_over_samples(
                            output_tmap, "atom"
                        )
                    else:
                        processed_output_tmap = output_tmap
                    output_quantities[ll_output_name] = processed_output_tmap
                if "features" in outputs:
                    features_options = outputs["features"]
                    if not features_options.per_atom:
                        processed_output_tmap = metatensor.torch.sum_over_samples(
                            output_tmap, "atom"
                        )
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
                output_tmap = metatensor.torch.sum_over_samples(output_tmap, "atom")
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



class FlashMD(torch.nn.Module):
    __supported_devices__ = ["cuda", "cpu"]
    __supported_dtypes__ = [torch.float64, torch.float32]
    __default_metadata__ = ModelMetadata(
        references={"architecture": ["https://arxiv.org/abs/2505.19350"]}
    )

    component_labels: dict[str, list[list[Labels]]]

    def __init__(self, model_hypers: dict, dataset_info: DatasetInfo) -> None:
        super().__init__()
        # checks on targets inside the RotationalAugmenter class in the trainer

        self.hypers = model_hypers
        self.dataset_info = dataset_info
        self.outputs = {
            "features": ModelOutput(unit="", per_atom=True)
        }  # the model is always capable of outputting the internal features

        self.heads = torch.nn.ModuleDict()
        self.head_types = self.hypers["heads"]
        self.last_layers = torch.nn.ModuleDict()
        self.output_shapes: Dict[str, Dict[str, List[int]]] = {}
        self.key_labels: Dict[str, Labels] = {}
        self.component_labels: Dict[str, List[List[Labels]]] = {}
        self.property_labels: Dict[str, List[Labels]] = {}
        for target_name, target_info in dataset_info.targets.items():
            self._add_output(target_name, target_info)

        self.is_direct = False
        self.is_separable = False
        self.kinetic_model = None

        dataset_info_for_nanopet = {}
        if model_hypers["hamiltonian"] == "direct":
            dataset_info_for_nanopet = DatasetInfo(
                length_unit=dataset_info.length_unit,
                atomic_types=dataset_info.atomic_types,
                targets={"mtt::delta_q": dataset_info.targets["mtt::delta_q"], "mtt::delta_p": dataset_info.targets["mtt::delta_p"]}
            )
            self.is_direct = True
        elif model_hypers["hamiltonian"] == "separable":
            dataset_info_for_nanopet = DatasetInfo(
                length_unit=dataset_info.length_unit,
                atomic_types=dataset_info.atomic_types,
                targets={"energy": get_energy_target_info({"quantity": "energy", "unit": ""})}
            )
            self.is_separable = True
            self.kinetic_model = ...
        elif model_hypers["hamiltonian"] == "generic":
            dataset_info_for_nanopet = DatasetInfo(
                length_unit=dataset_info.length_unit,
                atomic_types=dataset_info.atomic_types,
                targets={"mtt::hamiltonian": get_energy_target_info({"quantity": "energy", "unit": ""})}
            )
        else:
            raise ValueError()
        
        self.is_euler = False
        self.is_vv = False
        self.is_nsi = False
        if model_hypers["integrator"] == "euler":
            self.is_euler = True
        elif model_hypers["integrator"] == "vv":
            self.is_vv = True
        elif model_hypers["integrator"] == "nsi":
            self.is_nsi = True
        else:
            raise ValueError()

        self.model = PET(model_hypers, dataset_info) #FlashMDPET(model_hypers, dataset_info=dataset_info_for_nanopet)

        # scaler: this is also handled by the trainer at training time
        self.scaler = Scaler(hypers={}, dataset_info=dataset_info)
    
    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        device = systems[0].positions.device
        if list(self.key_labels.values())[0].device != device:
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

        assert len(outputs) == 2
        assert "mtt::delta_q" in outputs.keys()
        assert "mtt::delta_p" in outputs.keys()

        return_dict: Dict[str, TensorMap] = {}
        if self.is_direct:
            return_dict = self.model(systems, outputs, selected_atoms)
        else:
            qs = [system.positions.detach() for system in systems]
            ps = [system.get_data("momenta").block().values.squeeze(-1).detach() for system in systems]

            if self.is_euler:
                dHdqs, dHdps = self._get_H_derivatives(systems, qs, ps)
                qs = [q + dHdp for q, dHdp in zip(qs, dHdps)]
                ps = [p - dHdq for p, dHdq in zip(ps, dHdqs)]

            elif self.is_vv:
                dHdqs = self._get_dHdq(systems, qs, ps)
                ps = [p - 0.5*dHdq for p, dHdq in zip(ps, dHdqs)]

                dHdps = self._get_dHdp(systems, qs, ps)
                qs = [q + dHdp for q, dHdp in zip(qs, dHdps)]

                dHdqs = self._get_dHdq(systems, qs, ps)
                ps = [p - 0.5*dHdq for p, dHdq in zip(ps, dHdqs)]
            
            else:
                raise ValueError()

            delta_qs = [q - system.positions for q, system in zip(qs, systems)]
            delta_ps = [p - system.get_data("momenta").block().values.squeeze(-1) for p, system in zip(ps, systems)]

            delta_qs = torch.concatenate(delta_qs)
            delta_ps = torch.concatenate(delta_ps)

            system_indices = torch.concatenate(
                [
                    torch.full(
                        (len(system),),
                        i_system,
                        device=system.device,
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
                                device=system.device,
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

            for output_name in outputs:
                atomic_features = (delta_qs if output_name == "mtt::delta_q" else delta_ps)
                atomic_properties_by_block = [atomic_features]
                blocks = [
                    TensorBlock(
                        values=atomic_property.reshape([-1] + shape),
                        samples=sample_labels,
                        components=components,
                        properties=properties,
                    )
                    for atomic_property, shape, components, properties in zip(
                        atomic_properties_by_block,
                        self.output_shapes[output_name].values(),
                        self.component_labels[output_name],
                        self.property_labels[output_name],
                    )
                ]
                return_dict[output_name] = TensorMap(
                    keys=self.key_labels[output_name],
                    blocks=blocks,
                )

        if not self.training:
            # at evaluation, we also introduce the scaler
            return_dict = self.scaler(return_dict)

        return return_dict
        

    def _get_H_derivatives(self, systems: List[System], qs: List[torch.Tensor], ps: List[torch.Tensor]):
        tensors_with_grads = self._prepare_grad(qs + ps)
        qs = tensors_with_grads[:len(systems)]
        ps = tensors_with_grads[len(systems):]
        hamiltonians = self._evaluate_hamiltonian(systems, qs, ps)
        gradients = torch.autograd.grad(
            hamiltonians,
            qs + ps,
            torch.ones_like(hamiltonians),
            retain_graph=self.training,
            create_graph=self.training,
        )
        dHdqs = gradients[:len(systems)]
        dHdps = gradients[len(systems):]
        return dHdqs, dHdps
    
    def _get_dHdq(self, systems: List[System], qs: List[torch.Tensor], ps: List[torch.Tensor]):
        qs = self._prepare_grad(qs)
        hamiltonians = self._evaluate_hamiltonian(systems, qs, ps)
        dHdqs = torch.autograd.grad(
            hamiltonians,
            qs,
            torch.ones_like(hamiltonians),
            retain_graph=self.training,
            create_graph=self.training,
        )
        return dHdqs
    
    def _get_dHdp(self, systems: List[System], qs: List[torch.Tensor], ps: List[torch.Tensor]):
        ps = self._prepare_grad(ps)
        hamiltonians = self._evaluate_hamiltonian(systems, qs, ps)
        dHdps = torch.autograd.grad(
            hamiltonians,
            ps,
            torch.ones_like(hamiltonians),
            retain_graph=self.training,
            create_graph=self.training,
        )
        return dHdps

    def _prepare_grad(self, tensors: List[torch.Tensor]):
        for tensor in tensors:
            tensor.requires_grad_(True)
        return tensors

    def _evaluate_hamiltonian(self, systems: List[System], qs: List[torch.Tensor], ps: List[torch.Tensor]):
        # systems is used as a template here
        new_systems: List[System] = []
        for system, q, p in zip(systems, qs, ps):
            p_tmap = system.get_data("momenta")
            new_system = System(
                positions=q,
                types=system.types,
                cell=system.cell,
                pbc=system.pbc,
            )
            for nl_options in system.known_neighbor_lists():
                new_system.add_neighbor_list(nl_options, system.get_neighbor_list(nl_options))
            new_system.add_data(
                "momenta",
                TensorMap(
                    keys=p_tmap.keys,
                    blocks=[
                        TensorBlock(
                            values=p.unsqueeze(-1),
                            samples=p_tmap.block().samples,
                            components=p_tmap.block().components,
                            properties=p_tmap.block().properties,
                        )
                    ]
                )
            )
            new_systems.append(new_system)
        return self.model(
            new_systems,
            {"mtt::hamiltonian": ModelOutput()}
        )["mtt::hamiltonian"].block().values
    
    def _add_output(self, target_name: str, target_info: TargetInfo) -> None:
        # one output shape for each tensor block, grouped by target (i.e. tensormap)
        self.output_shapes[target_name] = {}
        for key, block in target_info.layout.items():
            dict_key = target_name
            for n, k in zip(key.names, key.values):
                dict_key += f"_{n}_{int(k)}"
            self.output_shapes[target_name][dict_key] = [
                len(comp.values) for comp in block.components
            ] + [len(block.properties.values)]

        self.outputs[target_name] = ModelOutput(
            quantity=target_info.quantity,
            unit=target_info.unit,
            per_atom=True,
        )
        if (
            target_name not in self.head_types  # default to MLP
            or self.head_types[target_name] == "mlp"
        ):
            self.heads[target_name] = torch.nn.Sequential(
                torch.nn.Linear(
                    self.hypers["d_pet"], 4 * self.hypers["d_pet"], bias=False
                ),
                torch.nn.SiLU(),
                torch.nn.Linear(
                    4 * self.hypers["d_pet"], self.hypers["d_pet"], bias=False
                ),
                torch.nn.SiLU(),
            )
        elif self.head_types[target_name] == "linear":
            self.heads[target_name] = torch.nn.Sequential()
        else:
            raise ValueError(
                f"Unsupported head type {self.head_types[target_name]} "
                f"for target {target_name}"
            )

        ll_features_name = (
            f"mtt::aux::{target_name.replace('mtt::', '')}_last_layer_features"
        )
        self.outputs[ll_features_name] = ModelOutput(per_atom=True)

        self.last_layers[target_name] = torch.nn.ModuleDict(
            {
                key: torch.nn.Linear(
                    self.hypers["d_pet"],
                    prod(shape),
                    bias=False,
                )
                for key, shape in self.output_shapes[target_name].items()
            }
        )

        self.key_labels[target_name] = target_info.layout.keys
        self.component_labels[target_name] = [
            block.components for block in target_info.layout.blocks()
        ]
        self.property_labels[target_name] = [
            block.properties for block in target_info.layout.blocks()
        ]