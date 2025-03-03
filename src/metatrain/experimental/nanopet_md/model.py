from math import prod
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
from ..nanopet.model import NanoPET

from ...utils.additive import ZBL, CompositionModel
from ...utils.data import DatasetInfo, TargetInfo
from ...utils.dtype import dtype_to_str
from ...utils.long_range import DummyLongRangeFeaturizer, LongRangeFeaturizer
from ...utils.metadata import append_metadata_references
from ...utils.scaler import Scaler
from .modules.encoder import Encoder
from .modules.nef import (
    edge_array_to_nef,
    get_corresponding_edges,
    get_nef_indices,
    nef_array_to_edges,
)
from .modules.radial_mask import get_radial_mask
from .modules.structures import concatenate_structures
from .modules.transformer import Transformer
from ...utils.data.target_info import get_energy_target_info

import warnings
warnings.filterwarnings("ignore")


class NanoPETMD(torch.nn.Module):
    """
    Re-implementation of the PET architecture (https://arxiv.org/pdf/2305.19302).

    The positions and atomic species are encoded into a high-dimensional space
    using a simple encoder. The resulting features (in NEF, or Node-Edge-Feature
    format*) are then processed by a series of transformer layers. This process is
    repeated for a number of message-passing layers, where features are exchanged
    between corresponding edges (ij and ji). The final representation is used to
    predict atomic properties through decoders named "heads".

    * NEF format: a three-dimensional tensor where the first dimension corresponds
    to the nodes, the second to the edges corresponding to the neighbors of the
    node (padded as different nodes might have different numbers of edges),
    and the third to the features.
    """

    __supported_devices__ = ["cuda", "cpu"]
    __supported_dtypes__ = [torch.float64, torch.float32]
    __default_metadata__ = ModelMetadata(
        references={"architecture": ["https://arxiv.org/abs/2305.19302v3"]}
    )

    component_labels: Dict[str, List[List[Labels]]]

    def __init__(self, model_hypers: Dict, dataset_info: DatasetInfo) -> None:
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
        if model_hypers["hamiltonian"] == "separable":
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

        self.model = NanoPET(model_hypers, dataset_info=dataset_info_for_nanopet)

        # scaler: this is also handled by the trainer at training time
        self.scaler = Scaler(model_hypers={}, dataset_info=dataset_info)

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


class KineticNN(torch.nn.Module):
    def __init__(self, model_hypers: Dict, dataset_info: DatasetInfo) -> None:
        super().__init__()
        self.d_pet = model_hypers["d_pet"]
        self.atomic_types = dataset_info.atomic_types
        self.register_buffer(
            "species_to_species_index",
            torch.full(
                (max(self.atomic_types) + 1,),
                -1,
            ),
        )
        for i, species in enumerate(self.atomic_types):
            self.species_to_species_index[species] = i

        self.type_embedding = torch.nn.Embedding(embedding_dim=self.d_pet, num_embeddings=4)
        self.momenta_embedding = torch.nn.Linear(3, self.d_pet, bias=False)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2*self.d_pet, 4 * self.d_pet, bias=False),
            torch.nn.SiLU(),
            torch.nn.Linear(4 * self.d_pet, 4 * self.d_pet, bias=False),
            torch.nn.SiLU(),
            torch.nn.Linear(4 * self.d_pet, 1, bias=False),
        )

    def forward(
        self,
        systems: List[System],
        momenta: List[torch.Tensor],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        device = systems[0].device

        momenta_concat = torch.concatenate(momenta)
        if selected_atoms is not None:
            raise NotImplementedError("Selected atoms not supported for kinetic energy")
        
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
        
        element_indices = torch.cat([self.species_to_species_index[system.types] for system in systems])
        element_features = self.type_embedding(element_indices)
        momenta_features = self.momenta_embedding(momenta_concat)
        features = torch.cat([element_features, momenta_features], dim=-1)
        kinetic_energy = self.mlp(features)

        return {"mtt::kinetic_energy": TensorMap(
            keys=Labels(
                names=["_"],
                values=torch.tensor([0], device=device).reshape(-1, 1),
            ),
            blocks=[
                TensorBlock(
                    values=kinetic_energy,
                    samples=sample_labels,
                    components=[],
                    properties=Labels(
                        names=["energy"],
                        values=torch.tensor([0], device=kinetic_energy.device).reshape(-1, 1),
                    ),
                )
            ],
        )}
