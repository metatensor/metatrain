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
from nanopet_neighbors import get_corresponding_edges, get_nef_indices

from ...utils.composition import apply_composition_contribution_samples
from ...utils.data import DatasetInfo
from ...utils.dtype import dtype_to_str
from ...utils.export import export
from .modules.encoder import Encoder
from .modules.nef import edge_array_to_nef, nef_array_to_edges
from .modules.radial_mask import get_radial_mask
from .modules.structures import concatenate_structures
from .modules.transformer import Transformer


class NanoPET(torch.nn.Module):

    __supported_devices__ = ["cuda", "cpu"]
    __supported_dtypes__ = [torch.float64, torch.float32]

    def __init__(self, model_hypers: Dict, dataset_info: DatasetInfo) -> None:
        super().__init__()
        self.hypers = model_hypers
        self.dataset_info = dataset_info
        self.new_outputs = list(dataset_info.targets.keys())
        self.atomic_types = dataset_info.atomic_types

        self.outputs = {
            key: ModelOutput(
                quantity=value.quantity,
                unit=value.unit,
                per_atom=True,
            )
            for key, value in dataset_info.targets.items()
        }

        # the model is always capable of outputting the last layer features
        self.outputs["mtt::aux::last_layer_features"] = ModelOutput(
            unit="unitless", per_atom=True
        )

        # creates a composition weight tensor that can be directly indexed by species,
        # this can be left as a tensor of zero or set from the outside using
        # set_composition_weights (recommended for better accuracy)
        n_outputs = len(self.outputs)
        self.register_buffer(
            "composition_weights",
            torch.zeros((n_outputs, max(self.atomic_types) + 1)),
        )
        # buffers cannot be indexed by strings (torchscript), so we create a single
        # tensor for all outputs. Due to this, we need to slice the tensor when we use
        # it and use the output name to select the correct slice via a dictionary
        self.output_to_index = {
            output_name: i for i, output_name in enumerate(self.outputs.keys())
        }

        self.encoder = Encoder(len(self.atomic_types), self.hypers["d_pet"])

        self.transformer = Transformer(
            self.hypers["d_pet"],
            4 * self.hypers["d_pet"],
            self.hypers["num_heads"],
            self.hypers["num_attention_layers"],
            self.hypers["mlp_dropout_rate"],
            self.hypers["attention_dropout_rate"],
        )

        self.num_mp_layers = self.hypers["num_gnn_layers"] - 1
        gnn_contractions = []
        gnn_transformers = []
        for _ in range(self.num_mp_layers):
            gnn_contractions.append(
                torch.nn.Linear(
                    2 * self.hypers["d_pet"], self.hypers["d_pet"], bias=False
                )
            )
            gnn_transformers.append(
                Transformer(
                    self.hypers["d_pet"],
                    4 * self.hypers["d_pet"],
                    self.hypers["num_heads"],
                    self.hypers["num_attention_layers"],
                    self.hypers["mlp_dropout_rate"],
                    self.hypers["attention_dropout_rate"],
                )
            )
        self.gnn_contractions = torch.nn.ModuleList(gnn_contractions)
        self.gnn_transformers = torch.nn.ModuleList(gnn_transformers)

        self.last_layer_feature_size = self.hypers["d_pet"]
        self.last_layers = torch.nn.ModuleDict(
            {
                output_name: torch.nn.Linear(self.hypers["d_pet"], 1, bias=False)
                for output_name in self.outputs.keys()
                if "mtt::aux::" not in output_name
            }
        )

        self.register_buffer(
            "species_to_species_index",
            torch.full(
                (max(self.atomic_types) + 1,),
                -1,
            ),
        )
        for i, species in enumerate(self.atomic_types):
            self.species_to_species_index[species] = i

    def restart(self, dataset_info: DatasetInfo) -> "NanoPET":
        # merge old and new dataset info
        merged_info = self.dataset_info.union(dataset_info)
        new_atomic_types = [
            at for at in merged_info.atomic_types if at not in self.atomic_types
        ]
        new_targets = merged_info.targets - self.dataset_info.targets

        if len(new_atomic_types) > 0:
            raise ValueError(
                f"New atomic types found in the dataset: {new_atomic_types}. "
                "The nanoPET model does not support adding new atomic types."
            )

        # register new outputs as new last layers
        for output_name in new_targets:
            self.add_output(output_name)

        self.dataset_info = merged_info
        self.atomic_types = sorted(self.atomic_types)

        for target_name, target in new_targets.items():
            self.outputs[target_name] = ModelOutput(
                quantity=target.quantity,
                unit=target.unit,
                per_atom=True,
            )
        self.new_outputs = list(new_targets.keys())

        return self

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        # Checks on systems (species) and outputs are done in the
        # MetatensorAtomisticModel wrapper

        (
            positions,
            centers,
            neighbors,
            species,
            segment_indices,
            edge_vectors,
            cell_shifts,
        ) = concatenate_structures(systems)
        max_edges_per_node = int(torch.max(torch.bincount(centers)))

        # Convert to NEF:
        nef_indices, nef_to_edges_neighbor, nef_mask = get_nef_indices(
            centers, len(positions), max_edges_per_node
        )

        # Get radial mask
        r = torch.sqrt(torch.sum(edge_vectors**2, dim=-1))
        radial_mask = get_radial_mask(r, 5.0, 3.0)

        # Element indices
        element_indices_nodes = self.species_to_species_index[species]
        element_indices_centers = element_indices_nodes[centers]
        element_indices_neighbors = element_indices_nodes[neighbors]

        # Send everything to NEF:
        edge_vectors = edge_array_to_nef(edge_vectors, nef_indices)
        radial_mask = edge_array_to_nef(
            radial_mask, nef_indices, nef_mask, fill_value=0.0
        )
        element_indices_centers = edge_array_to_nef(
            element_indices_centers, nef_indices
        )
        element_indices_neighbors = edge_array_to_nef(
            element_indices_neighbors, nef_indices
        )

        features = {
            "cartesian": edge_vectors,
            "center": element_indices_centers,
            "neighbor": element_indices_neighbors,
        }

        # Encode
        features = self.encoder(features)

        # Transformer
        features = self.transformer(features, radial_mask)

        # GNN
        if self.num_mp_layers > 0:
            corresponding_edges = get_corresponding_edges(
                torch.concatenate(
                    [centers.unsqueeze(-1), neighbors.unsqueeze(-1), cell_shifts],
                    dim=-1,
                )
            )
            for contraction, transformer in zip(
                self.gnn_contractions, self.gnn_transformers
            ):
                new_features = nef_array_to_edges(
                    features, centers, nef_to_edges_neighbor
                )
                corresponding_new_features = new_features[corresponding_edges]
                new_features = torch.concatenate(
                    [new_features, corresponding_new_features], dim=-1
                )
                new_features = contraction(new_features)
                new_features = edge_array_to_nef(new_features, nef_indices)
                new_features = transformer(new_features, radial_mask)
                features = features + new_features

        edge_features = features * radial_mask[:, :, None]
        node_features = torch.sum(edge_features, dim=1)

        return_dict: Dict[str, TensorMap] = {}

        # output the hidden features, if requested:
        if "mtt::aux::last_layer_features" in outputs:
            last_layer_feature_tmap = TensorMap(
                keys=Labels(
                    names=["_"],
                    values=torch.tensor([[0]], device=node_features.device),
                ),
                blocks=[
                    TensorBlock(
                        values=node_features,
                        samples=Labels(
                            names=["system", "atom"],
                            values=torch.stack(
                                [
                                    segment_indices,
                                    torch.concatenate(
                                        [
                                            torch.arange(
                                                len(system),
                                                device=node_features.device,
                                            )
                                            for system in systems
                                        ],
                                    ),
                                ],
                                dim=1,
                            ),
                        ),
                        components=[],
                        properties=Labels(
                            names=["property"],
                            values=torch.arange(
                                node_features.shape[-1], device=node_features.device
                            ).reshape(-1, 1),
                        ),
                    )
                ],
            )
            last_layer_features_options = outputs["mtt::aux::last_layer_features"]
            if last_layer_features_options.per_atom:
                return_dict["mtt::aux::last_layer_features"] = last_layer_feature_tmap
            else:
                return_dict["mtt::aux::last_layer_features"] = (
                    metatensor.torch.sum_over_samples(last_layer_feature_tmap, ["atom"])
                )

        atomic_energies_tmap_dict: Dict[str, TensorMap] = {}
        for output_name, last_layer in self.last_layers.items():
            if output_name in outputs:
                atomic_energies = last_layer(node_features)
                atomic_energies_tmap_dict[output_name] = TensorMap(
                    keys=Labels(
                        names=["_"],
                        values=torch.tensor([[0]], device=node_features.device),
                    ),
                    blocks=[
                        TensorBlock(
                            values=atomic_energies,
                            samples=Labels(
                                names=["system", "atom", "center_type"],
                                values=torch.stack(
                                    [
                                        segment_indices,
                                        torch.concatenate(
                                            [
                                                torch.arange(
                                                    len(system),
                                                    device=atomic_energies.device,
                                                )
                                                for system in systems
                                            ],
                                        ),
                                        species,
                                    ],
                                    dim=1,
                                ),
                            ),
                            components=[],
                            properties=Labels(
                                names=["energy"],
                                values=torch.tensor(
                                    [[0]], device=atomic_energies.device
                                ),
                            ),
                        )
                    ],
                )

        for output_name, tmap in atomic_energies_tmap_dict.items():
            atomic_energies_tmap_dict[output_name] = (
                apply_composition_contribution_samples(
                    tmap, self.composition_weights[self.output_to_index[output_name]]
                )
            )

        if selected_atoms is not None:
            for output_name, tmap in atomic_energies_tmap_dict.items():
                atomic_energies_tmap_dict[output_name] = metatensor.torch.slice(
                    tmap, axis="samples", labels=selected_atoms
                )

        for output_name, atomic_energy in atomic_energies_tmap_dict.items():
            if outputs[output_name].per_atom:
                # this operation should just remove the center_type label
                return_dict[output_name] = metatensor.torch.remove_dimension(
                    atomic_energy, axis="samples", name="center_type"
                )
            else:
                return_dict[output_name] = metatensor.torch.sum_over_samples(
                    atomic_energy, ["atom", "center_type"]
                )

        return return_dict

    def requested_neighbor_lists(
        self,
    ) -> List[NeighborListOptions]:
        return [
            NeighborListOptions(
                cutoff=self.hypers["cutoff"],
                full_list=True,
            )
        ]

    @classmethod
    def load_checkpoint(cls, path: Union[str, Path]) -> "NanoPET":

        # Load the checkpoint
        checkpoint = torch.load(path)
        model_hypers = checkpoint["model_hypers"]
        model_state_dict = checkpoint["model_state_dict"]

        # Create the model
        model = cls(**model_hypers)
        dtype = next(iter(model_state_dict.values())).dtype
        model.to(dtype).load_state_dict(model_state_dict)

        return model

    def export(self) -> MetatensorAtomisticModel:
        dtype = next(self.parameters()).dtype
        if dtype not in self.__supported_dtypes__:
            raise ValueError(f"unsupported dtype {self.dtype} for NanoPET")

        capabilities = ModelCapabilities(
            outputs=self.outputs,
            atomic_types=self.atomic_types,
            interaction_range=self.hypers["cutoff"] * self.hypers["num_gnn_layers"],
            length_unit=self.dataset_info.length_unit,
            supported_devices=self.__supported_devices__,
            dtype=dtype_to_str(dtype),
        )

        return export(model=self, model_capabilities=capabilities)

    def set_composition_weights(
        self,
        output_name: str,
        input_composition_weights: torch.Tensor,
        atomic_types: List[int],
    ) -> None:
        """Set the composition weights for a given output."""
        # all species that are not present retain their weight of zero
        self.composition_weights[self.output_to_index[output_name]][  # type: ignore
            atomic_types
        ] = input_composition_weights.to(
            dtype=self.composition_weights.dtype,  # type: ignore
            device=self.composition_weights.device,  # type: ignore
        )
