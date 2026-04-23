import copy
import logging
from typing import Any, Dict, List, Literal, Optional, Tuple

import metatensor.torch as mts
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import ModelMetadata, ModelOutput, System
from torch.nn import Linear, ModuleDict, ModuleList, Sequential, SiLU

from metatrain.pet.model import (
    PET,
    get_last_layer_features_name,
    process_non_conservative_stress,
)
from metatrain.pet.modules.structures import concatenate_structures
from metatrain.soap_bpnn.modules.tensor_basis import TensorBasis
from metatrain.utils.data import DatasetInfo, TargetInfo
from metatrain.utils.sum_over_atoms import sum_over_atoms

from .documentation import ModelHypers


def _basis_gram_chunk_penalty(tensor_basis: torch.Tensor) -> torch.Tensor:
    """Regularize each full VectorBasis-sized chunk as its own local frame."""
    num_components = tensor_basis.shape[1]
    basis_width = tensor_basis.shape[2]
    if num_components <= 1 or basis_width < num_components:
        return torch.tensor(
            0.0, device=tensor_basis.device, dtype=tensor_basis.dtype
        )

    num_full_chunks = basis_width // num_components
    if num_full_chunks == 0:
        return torch.tensor(
            0.0, device=tensor_basis.device, dtype=tensor_basis.dtype
        )

    penalties: List[torch.Tensor] = []
    eye = torch.eye(
        num_components, device=tensor_basis.device, dtype=tensor_basis.dtype
    ).unsqueeze(0)
    for chunk_index in range(num_full_chunks):
        start = chunk_index * num_components
        end = start + num_components
        chunk = tensor_basis[:, :, start:end]
        denom = chunk.pow(2).sum(dim=1, keepdim=True).clamp_min(1e-12).sqrt()
        normalized_chunk = chunk / denom
        gram = torch.einsum("scb,scd->sbd", normalized_chunk, normalized_chunk)
        penalties.append((gram - eye).pow(2).mean())

    return torch.stack(penalties).mean()


def _add_l1_species_dependent_vector(
    tensor_basis_hypers: Dict[str, Any]
) -> bool:
    return bool(tensor_basis_hypers.get("add_l1_species_dependent_vector", False))


def _l1_species_dependent_vector_soap(
    tensor_basis_hypers: Dict[str, Any]
) -> dict[str, Any]:
    if "l1_species_dependent_vector_soap" in tensor_basis_hypers:
        return copy.deepcopy(tensor_basis_hypers["l1_species_dependent_vector_soap"])
    return copy.deepcopy(tensor_basis_hypers["soap"])


def _extra_l1_vector_basis_branches(
    tensor_basis_hypers: Dict[str, Any]
) -> list[dict[str, Any]]:
    branches = copy.deepcopy(
        tensor_basis_hypers.get("extra_l1_vector_basis_branches", [])
    )
    if branches:
        return branches
    if tensor_basis_hypers.get("add_l1_extra_vector_basis", False):
        return [
            copy.deepcopy(
                tensor_basis_hypers.get(
                    "l1_extra_vector_basis_soap", tensor_basis_hypers["soap"]
                )
            )
        ]
    return []


def normalize_by_volume(tensor_map: TensorMap, systems: List[System]) -> TensorMap:
    """Normalize reconstructed outputs by per-system cell volume."""
    volumes = torch.stack([torch.abs(torch.det(system.cell)) for system in systems])
    volumes[volumes == 0.0] = torch.inf

    new_blocks = torch.jit.annotate(List[TensorBlock], [])
    for block in tensor_map.blocks():
        system_samples = block.samples.column("system").to(torch.long)
        block_volumes = volumes.to(
            device=block.values.device, dtype=block.values.dtype
        )[system_samples]
        for _ in range(block.values.ndim - 1):
            block_volumes = block_volumes.unsqueeze(-1)
        new_blocks.append(
            TensorBlock(
                values=block.values / block_volumes,
                samples=block.samples,
                components=block.components,
                properties=block.properties,
            )
        )

    return TensorMap(keys=tensor_map.keys, blocks=new_blocks)


class EPET(PET):
    __checkpoint_version__ = 1
    __default_metadata__ = ModelMetadata(
        references={
            "architecture": [
                "PET backbone: https://arxiv.org/abs/2305.19302v3",
                "SOAP tensor basis: https://link.aps.org/doi/10.1103/PhysRevLett.98.146401",
            ]
        }
    )

    output_shapes: Dict[str, Dict[str, List[int]]]
    key_labels: Dict[str, Labels]
    property_labels: Dict[str, List[Labels]]
    component_labels: Dict[str, List[List[Labels]]]
    target_names: List[str]
    scalar_target_names: List[str]
    spherical_target_names: List[str]
    volume_normalized_target_names: List[str]
    tensor_basis_legacy: bool
    coefficient_shapes: Dict[str, Dict[str, Tuple[int, int]]]
    target_head_keys: Dict[str, List[str]]
    block_to_head_key: Dict[str, Dict[str, str]]
    block_irrep_keys: Dict[str, Dict[str, str]]
    latest_diagnostics: Dict[str, Dict[str, float]]
    shared_head_selectors: Dict[str, str]
    top_level_shared_head_selectors: Dict[str, str]

    def __init__(self, hypers: ModelHypers, dataset_info: DatasetInfo) -> None:
        self.e_pet_hypers = copy.deepcopy(hypers)
        self.tensor_basis_hypers = copy.deepcopy(hypers["tensor_basis_defaults"])
        self.tensor_basis_hypers.setdefault("extra_l1_vector_basis_branches", [])
        self.tensor_basis_hypers.setdefault("add_l1_species_dependent_vector", False)
        self.tensor_basis_hypers.setdefault(
            "l1_species_dependent_vector_soap",
            copy.deepcopy(self.tensor_basis_hypers["soap"]),
        )
        self.e_pet_hypers["tensor_basis_defaults"] = copy.deepcopy(
            self.tensor_basis_hypers
        )
        self.tensor_basis_legacy = bool(self.tensor_basis_hypers["legacy"])
        self.irrep_head_groups_config = copy.deepcopy(
            hypers.get("irrep_head_groups", {})
        )
        self.shared_head_groups_config = copy.deepcopy(
            hypers.get("shared_head_groups", {})
        )
        self.volume_normalized_target_names = list(
            hypers.get("volume_normalized_targets", [])
        )
        self.scalar_target_names = []
        self.spherical_target_names = []
        self.basis_calculators: Optional[ModuleDict] = None
        self.coefficient_shapes = {}
        self.target_head_keys = {}
        self.block_to_head_key = {}
        self.block_irrep_keys = {}
        self.top_level_shared_head_selectors = (
            self._validate_and_build_shared_head_groups(dataset_info)
        )
        self.shared_head_selectors = copy.deepcopy(
            self.top_level_shared_head_selectors
        )
        self._last_spherical_coefficient_penalty = torch.tensor(0.0)
        self._last_basis_gram_penalty = torch.tensor(0.0)
        self.latest_diagnostics = {}
        super().__init__(copy.deepcopy(hypers["pet"]), dataset_info)
        self.shared_head_groups_config = copy.deepcopy(
            hypers.get("shared_head_groups", {})
        )
        self.shared_head_selectors = copy.deepcopy(
            self.top_level_shared_head_selectors
        )
        self._validate_irrep_head_groups_against_dataset(dataset_info)

        unknown_volume_normalized_targets = set(self.volume_normalized_target_names) - set(
            self.target_names
        )
        if unknown_volume_normalized_targets:
            raise ValueError(
                "Unknown volume-normalized target names: "
                f"{sorted(unknown_volume_normalized_targets)}. Known targets are: "
                f"{sorted(self.target_names)}."
            )

    @staticmethod
    def _block_dict_key(target_name: str, key: Labels) -> str:
        dict_key = target_name
        for name, value in zip(key.names, key.values, strict=True):
            dict_key += f"_{name}_{int(value)}"
        return dict_key

    @staticmethod
    def _irrep_key(key: Labels) -> str:
        return f"{int(key[0])},{int(key[1])}"

    @staticmethod
    def _shared_selector(target_name: str, irrep_key: Optional[str] = None) -> str:
        if irrep_key is None:
            return target_name
        return f"{target_name}[{irrep_key}]"

    @staticmethod
    def _parse_shared_selector(selector: str) -> Tuple[str, Optional[str]]:
        if "[" not in selector and "]" not in selector:
            return selector, None
        if selector.count("[") != 1 or not selector.endswith("]"):
            raise ValueError(
                f"Invalid shared_head_groups selector '{selector}'. Expected "
                '"target" or "target[o3_lambda,o3_sigma]".'
            )
        target_name, irrep_key = selector[:-1].split("[", 1)
        if target_name == "" or irrep_key == "":
            raise ValueError(
                f"Invalid shared_head_groups selector '{selector}'. Expected "
                '"target" or "target[o3_lambda,o3_sigma]".'
            )
        return target_name, irrep_key

    def _validate_and_build_shared_head_groups(
        self, dataset_info: DatasetInfo
    ) -> Dict[str, str]:
        if not self.shared_head_groups_config:
            return {}

        known_targets = set(dataset_info.targets)
        selector_to_head_key: Dict[str, str] = {}
        for group_name, selectors in self.shared_head_groups_config.items():
            head_key = f"shared__{group_name}"
            for selector in selectors:
                target_name, irrep_key = self._parse_shared_selector(selector)
                if target_name not in known_targets:
                    raise ValueError(
                        "Unknown targets in shared_head_groups: "
                        f"{sorted({target_name})}. Known targets are: "
                        f"{sorted(known_targets)}."
                    )

                target_info = dataset_info.targets[target_name]
                if target_info.is_scalar:
                    if irrep_key is not None:
                        raise ValueError(
                            "Scalar selectors cannot include an irrep suffix in "
                            f"shared_head_groups: {selector!r}."
                        )
                    canonical_selector = self._shared_selector(target_name)
                elif target_info.is_spherical:
                    if irrep_key is None:
                        raise ValueError(
                            "Spherical selectors in shared_head_groups must include "
                            f"an explicit irrep suffix: {selector!r}."
                        )
                    known_irrep_keys = {
                        self._irrep_key(key) for key, _ in target_info.layout.items()
                    }
                    if irrep_key not in known_irrep_keys:
                        raise ValueError(
                            f"Unknown irrep key '{irrep_key}' for target "
                            f"'{target_name}' in shared_head_groups. Known irreps "
                            f"are: {sorted(known_irrep_keys)}."
                        )
                    if (
                        target_name in self.irrep_head_groups_config
                        and irrep_key in self.irrep_head_groups_config[target_name]
                    ):
                        raise ValueError(
                            "Spherical selectors cannot appear in both "
                            "shared_head_groups and irrep_head_groups: "
                            f"{selector!r}."
                        )
                    canonical_selector = self._shared_selector(target_name, irrep_key)
                else:
                    raise ValueError(
                        "Only scalar and spherical targets can appear in "
                        f"shared_head_groups; target '{target_name}' is unsupported."
                    )

                if canonical_selector in selector_to_head_key:
                    existing_head_key = selector_to_head_key[canonical_selector]
                    if existing_head_key != head_key:
                        raise ValueError(
                            "A selector may belong to only one shared_head_groups "
                            f"entry: {canonical_selector!r}."
                        )
                    continue

                selector_to_head_key[canonical_selector] = head_key

        return selector_to_head_key

    def _validate_irrep_head_groups_against_dataset(
        self, dataset_info: DatasetInfo
    ) -> None:
        if not self.irrep_head_groups_config:
            return

        known_targets = set(dataset_info.targets)
        unknown_targets = set(self.irrep_head_groups_config) - known_targets
        if unknown_targets:
            raise ValueError(
                "Unknown targets in irrep_head_groups: "
                f"{sorted(unknown_targets)}. Known targets are: {sorted(known_targets)}."
            )

        scalar_targets = sorted(
            target_name
            for target_name in self.irrep_head_groups_config
            if dataset_info.targets[target_name].is_scalar
        )
        if scalar_targets:
            raise ValueError(
                "Scalar targets cannot appear in irrep_head_groups: "
                f"{scalar_targets}."
            )

        non_spherical_targets = sorted(
            target_name
            for target_name in self.irrep_head_groups_config
            if not dataset_info.targets[target_name].is_spherical
        )
        if non_spherical_targets:
            raise ValueError(
                "Only spherical targets can appear in irrep_head_groups: "
                f"{non_spherical_targets}."
            )

    def _add_output(self, target_name: str, target_info: TargetInfo) -> None:
        extra_l1_vector_basis_branches = _extra_l1_vector_basis_branches(
            self.tensor_basis_hypers
        )
        add_l1_species_dependent_vector = _add_l1_species_dependent_vector(
            self.tensor_basis_hypers
        )
        l1_species_dependent_vector_soap = _l1_species_dependent_vector_soap(
            self.tensor_basis_hypers
        )

        if target_info.is_scalar:
            self.scalar_target_names.append(target_name)
            super()._add_output(target_name, target_info)
            head_key = self.top_level_shared_head_selectors.get(
                self._shared_selector(target_name), target_name
            )
            self.target_head_keys[target_name] = [head_key]
            if head_key != target_name:
                if head_key in self.node_heads:
                    del self.node_heads[target_name]
                    del self.edge_heads[target_name]
                else:
                    self.node_heads[head_key] = self.node_heads[target_name]
                    del self.node_heads[target_name]
                    self.edge_heads[head_key] = self.edge_heads[target_name]
                    del self.edge_heads[target_name]
            return

        if not target_info.is_spherical:
            raise ValueError(
                "experimental.e_pet supports only scalar and spherical targets in v1."
            )

        self.spherical_target_names.append(target_name)
        if self.basis_calculators is None:
            self.basis_calculators = ModuleDict({})

        output_layout = target_info.layout
        configured_groups = self.irrep_head_groups_config.get(target_name, {})

        self.output_shapes[target_name] = {}
        self.coefficient_shapes[target_name] = {}
        self.basis_calculators[target_name] = ModuleDict({})
        self.block_to_head_key[target_name] = {}
        self.block_irrep_keys[target_name] = {}
        self.target_head_keys[target_name] = []

        seen_irrep_keys: set[str] = set()
        group_name_to_internal_key: dict[str, str] = {}

        for block_index, (key, block) in enumerate(output_layout.items()):
            dict_key = self._block_dict_key(target_name, key)
            irrep_key = self._irrep_key(key)
            seen_irrep_keys.add(irrep_key)
            self.block_irrep_keys[target_name][dict_key] = irrep_key

            shared_selector = self._shared_selector(target_name, irrep_key)
            if shared_selector in self.top_level_shared_head_selectors:
                head_key = self.top_level_shared_head_selectors[shared_selector]
            elif irrep_key in configured_groups:
                configured_group = configured_groups[irrep_key]
                if configured_group not in group_name_to_internal_key:
                    group_name_to_internal_key[configured_group] = (
                        f"{target_name}__group__{len(group_name_to_internal_key)}"
                    )
                head_key = group_name_to_internal_key[configured_group]
            else:
                head_key = f"{target_name}__block__{block_index}"

            self.block_to_head_key[target_name][dict_key] = head_key
            if head_key not in self.target_head_keys[target_name]:
                self.target_head_keys[target_name].append(head_key)

            self.output_shapes[target_name][dict_key] = [
                len(component.values) for component in block.components
            ] + [len(block.properties.values)]

            num_components = len(block.components[0])
            if (
                num_components == 3
                and int(key[0]) == 1
                and int(key[1]) == 1
                and (
                    len(extra_l1_vector_basis_branches) > 0
                    or add_l1_species_dependent_vector
                )
            ):
                basis_size = (
                    (1 + len(extra_l1_vector_basis_branches)) * num_components
                    + (1 if add_l1_species_dependent_vector else 0)
                )
            elif num_components in (1, 3):
                basis_size = num_components
            elif self.tensor_basis_hypers["add_lambda_basis"]:
                basis_size = 2 * num_components
            else:
                basis_size = num_components

            self.coefficient_shapes[target_name][dict_key] = (
                len(block.properties.values),
                basis_size,
            )

            self.basis_calculators[target_name][dict_key] = TensorBasis(
                self.atomic_types,
                self.tensor_basis_hypers["soap"],
                int(key[0]),
                int(key[1]),
                add_lambda_basis=self.tensor_basis_hypers["add_lambda_basis"],
                legacy=self.tensor_basis_hypers["legacy"],
                add_l1_species_dependent_vector=add_l1_species_dependent_vector,
                l1_species_dependent_vector_soap=l1_species_dependent_vector_soap,
                add_l1_extra_vector_basis=self.tensor_basis_hypers.get(
                    "add_l1_extra_vector_basis", False
                ),
                l1_extra_vector_basis_soap=self.tensor_basis_hypers.get(
                    "l1_extra_vector_basis_soap",
                    self.tensor_basis_hypers["soap"],
                ),
                extra_l1_vector_basis_soaps=extra_l1_vector_basis_branches,
            )

        unknown_irrep_keys = set(configured_groups) - seen_irrep_keys
        if unknown_irrep_keys:
            raise ValueError(
                f"Unknown irrep keys for target '{target_name}' in irrep_head_groups: "
                f"{sorted(unknown_irrep_keys)}. Known irreps are: "
                f"{sorted(seen_irrep_keys)}."
            )

        self.outputs[target_name] = ModelOutput(
            quantity=target_info.quantity,
            unit=target_info.unit,
            per_atom=True,
            description=target_info.description,
        )

        for head_key in self.target_head_keys[target_name]:
            if head_key not in self.node_heads:
                self.node_heads[head_key] = ModuleList(
                    [
                        Sequential(
                            Linear(self.d_node, self.d_head),
                            SiLU(),
                            Linear(self.d_head, self.d_head),
                            SiLU(),
                        )
                        for _ in range(self.num_readout_layers)
                    ]
                )
            if head_key not in self.edge_heads:
                self.edge_heads[head_key] = ModuleList(
                    [
                        Sequential(
                            Linear(self.d_pet, self.d_head),
                            SiLU(),
                            Linear(self.d_head, self.d_head),
                            SiLU(),
                        )
                        for _ in range(self.num_readout_layers)
                    ]
                )

        self.node_last_layers[target_name] = ModuleList(
            [
                ModuleDict(
                    {
                        key: Linear(
                            self.d_head,
                            num_properties * basis_size,
                            bias=True,
                        )
                        for key, (num_properties, basis_size) in self.coefficient_shapes[
                            target_name
                        ].items()
                    }
                )
                for _ in range(self.num_readout_layers)
            ]
        )
        self.edge_last_layers[target_name] = ModuleList(
            [
                ModuleDict(
                    {
                        key: Linear(
                            self.d_head,
                            num_properties * basis_size,
                            bias=True,
                        )
                        for key, (num_properties, basis_size) in self.coefficient_shapes[
                            target_name
                        ].items()
                    }
                )
                for _ in range(self.num_readout_layers)
            ]
        )

        self.last_layer_parameter_names[target_name] = []
        for layer_index in range(self.num_readout_layers):
            for key in self.coefficient_shapes[target_name].keys():
                self.last_layer_parameter_names[target_name].append(
                    f"node_last_layers.{target_name}.{layer_index}.{key}.weight"
                )
                self.last_layer_parameter_names[target_name].append(
                    f"edge_last_layers.{target_name}.{layer_index}.{key}.weight"
                )

        ll_features_name = get_last_layer_features_name(target_name)
        self.outputs[ll_features_name] = ModelOutput(
            per_atom=True, description=f"last layer features for {target_name}"
        )
        self.key_labels[target_name] = output_layout.keys
        self.component_labels[target_name] = [
            block.components for block in output_layout.blocks()
        ]
        self.property_labels[target_name] = [
            block.properties for block in output_layout.blocks()
        ]

    def get_regularization_loss(self) -> torch.Tensor:
        return self._last_spherical_coefficient_penalty

    def get_basis_gram_loss(self) -> torch.Tensor:
        return self._last_basis_gram_penalty

    def _compute_interatomic_vectors(
        self, systems: List[System]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            positions,
            centers,
            neighbors,
            species,
            cells,
            cell_shifts,
            system_indices,
            _,
        ) = concatenate_structures(systems, self.requested_nl)

        if len(cells) == 1:
            cell_contributions = cell_shifts.to(cells.dtype) @ cells[0]
        else:
            cell_contributions = torch.einsum(
                "ab, abc -> ac",
                cell_shifts.to(cells.dtype),
                cells[system_indices[centers]],
            )
        interatomic_vectors = (
            positions[neighbors] - positions[centers] + cell_contributions
        )
        return interatomic_vectors, centers, neighbors, species, system_indices

    def _tensor_basis_species(self, species: torch.Tensor) -> torch.Tensor:
        if self.tensor_basis_legacy:
            return species
        return self.species_to_species_index[species]

    def _get_output_last_layer_features(
        self,
        node_last_layer_features_dict: Dict[str, List[torch.Tensor]],
        edge_last_layer_features_dict: Dict[str, List[torch.Tensor]],
        cutoff_factors: torch.Tensor,
        selected_atoms: Optional[Labels],
        sample_labels: Labels,
        requested_outputs: Dict[str, ModelOutput],
    ) -> Dict[str, TensorMap]:
        last_layer_features_outputs: Dict[str, TensorMap] = {}

        for output_name, output_options in requested_outputs.items():
            if not (
                output_name.startswith("mtt::aux::")
                and output_name.endswith("_last_layer_features")
            ):
                continue

            base_name = output_name.replace("mtt::aux::", "").replace(
                "_last_layer_features", ""
            )
            if f"mtt::{base_name}" in self.outputs:
                base_name = f"mtt::{base_name}"

            feature_keys = torch.jit.annotate(List[str], [])
            if base_name in self.target_head_keys:
                feature_keys = self.target_head_keys[base_name]
            if len(feature_keys) == 0:
                continue

            stacked_features: List[torch.Tensor] = []
            for feature_key in feature_keys:
                for readout_index in range(len(node_last_layer_features_dict[feature_key])):
                    stacked_features.append(
                        node_last_layer_features_dict[feature_key][readout_index]
                    )
                    stacked_features.append(
                        (
                            edge_last_layer_features_dict[feature_key][readout_index]
                            * cutoff_factors[:, :, None]
                        ).sum(dim=1)
                    )

            values = torch.cat(stacked_features, dim=1)
            block = TensorBlock(
                values=values,
                samples=sample_labels,
                components=[],
                properties=Labels(
                    names=["feature"],
                    values=torch.arange(
                        values.shape[-1], device=values.device
                    ).reshape(-1, 1),
                    assume_unique=True,
                ),
            )
            last_layer_feature_tmap = TensorMap(keys=self.single_label, blocks=[block])
            if selected_atoms is not None:
                last_layer_feature_tmap = mts.slice(
                    last_layer_feature_tmap,
                    axis="samples",
                    selection=selected_atoms,
                )
            if output_options.per_atom:
                last_layer_features_outputs[output_name] = last_layer_feature_tmap
            else:
                last_layer_features_outputs[output_name] = sum_over_atoms(
                    last_layer_feature_tmap
                )

        return last_layer_features_outputs

    def _calculate_atomic_predictions(
        self,
        node_last_layer_features_dict: Dict[str, List[torch.Tensor]],
        edge_last_layer_features_dict: Dict[str, List[torch.Tensor]],
        padding_mask: torch.Tensor,
        cutoff_factors: torch.Tensor,
        outputs: Dict[str, ModelOutput],
    ) -> Tuple[
        Dict[str, List[List[torch.Tensor]]], Dict[str, List[List[torch.Tensor]]]
    ]:
        node_atomic_predictions_dict: Dict[str, List[List[torch.Tensor]]] = {}
        edge_atomic_predictions_dict: Dict[str, List[List[torch.Tensor]]] = {}

        for output_name, node_last_layers in self.node_last_layers.items():
            if output_name in outputs:
                if output_name in self.scalar_target_names:
                    node_atomic_predictions_dict[output_name] = torch.jit.annotate(
                        List[List[torch.Tensor]], []
                    )
                    head_key = self.target_head_keys[output_name][0]
                    for i, node_last_layer in enumerate(node_last_layers):
                        node_last_layer_features = node_last_layer_features_dict[
                            head_key
                        ][i]
                        node_atomic_predictions_by_block: List[torch.Tensor] = []
                        for node_last_layer_by_block in node_last_layer.values():
                            node_atomic_predictions_by_block.append(
                                node_last_layer_by_block(node_last_layer_features)
                            )
                        node_atomic_predictions_dict[output_name].append(
                            node_atomic_predictions_by_block
                        )
                else:
                    node_atomic_predictions_dict[output_name] = torch.jit.annotate(
                        List[List[torch.Tensor]], []
                    )
                    for i, node_last_layer in enumerate(node_last_layers):
                        node_atomic_predictions_by_block: List[torch.Tensor] = []
                        for block_key, node_last_layer_by_block in node_last_layer.items():
                            head_key = self.block_to_head_key[output_name][block_key]
                            node_last_layer_features = node_last_layer_features_dict[
                                head_key
                            ][i]
                            node_atomic_predictions_by_block.append(
                                node_last_layer_by_block(node_last_layer_features)
                            )
                        node_atomic_predictions_dict[output_name].append(
                            node_atomic_predictions_by_block
                        )

        for output_name, edge_last_layers in self.edge_last_layers.items():
            if output_name in outputs:
                if output_name in self.scalar_target_names:
                    edge_atomic_predictions_dict[output_name] = torch.jit.annotate(
                        List[List[torch.Tensor]], []
                    )
                    head_key = self.target_head_keys[output_name][0]
                    for i, edge_last_layer in enumerate(edge_last_layers):
                        edge_last_layer_features = edge_last_layer_features_dict[
                            head_key
                        ][i]
                        edge_atomic_predictions_by_block: List[torch.Tensor] = []
                        for edge_last_layer_by_block in edge_last_layer.values():
                            edge_atomic_predictions = edge_last_layer_by_block(
                                edge_last_layer_features
                            )
                            expanded_padding_mask = padding_mask[..., None].repeat(
                                1, 1, edge_atomic_predictions.shape[2]
                            )
                            edge_atomic_predictions = torch.where(
                                ~expanded_padding_mask, 0.0, edge_atomic_predictions
                            )
                            edge_atomic_predictions_by_block.append(
                                (
                                    edge_atomic_predictions
                                    * cutoff_factors[:, :, None]
                                ).sum(dim=1)
                            )
                        edge_atomic_predictions_dict[output_name].append(
                            edge_atomic_predictions_by_block
                        )
                else:
                    edge_atomic_predictions_dict[output_name] = torch.jit.annotate(
                        List[List[torch.Tensor]], []
                    )
                    for i, edge_last_layer in enumerate(edge_last_layers):
                        edge_atomic_predictions_by_block: List[torch.Tensor] = []
                        for block_key, edge_last_layer_by_block in edge_last_layer.items():
                            head_key = self.block_to_head_key[output_name][block_key]
                            edge_last_layer_features = edge_last_layer_features_dict[
                                head_key
                            ][i]
                            edge_atomic_predictions = edge_last_layer_by_block(
                                edge_last_layer_features
                            )
                            expanded_padding_mask = padding_mask[..., None].repeat(
                                1, 1, edge_atomic_predictions.shape[2]
                            )
                            edge_atomic_predictions = torch.where(
                                ~expanded_padding_mask, 0.0, edge_atomic_predictions
                            )
                            edge_atomic_predictions_by_block.append(
                                (
                                    edge_atomic_predictions
                                    * cutoff_factors[:, :, None]
                                ).sum(dim=1)
                            )
                        edge_atomic_predictions_dict[output_name].append(
                            edge_atomic_predictions_by_block
                        )

        return node_atomic_predictions_dict, edge_atomic_predictions_dict

    def _get_output_atomic_predictions(
        self,
        systems: List[System],
        node_atomic_predictions_dict: Dict[str, List[List[torch.Tensor]]],
        edge_atomic_predictions_dict: Dict[str, List[List[torch.Tensor]]],
        edge_vectors: torch.Tensor,
        system_indices: torch.Tensor,
        sample_labels: Labels,
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
    ) -> Dict[str, TensorMap]:
        atomic_predictions_tmap_dict: Dict[str, TensorMap] = {}
        coefficient_penalties: List[torch.Tensor] = []
        basis_gram_penalties: List[torch.Tensor] = []
        diagnostics: Dict[str, Dict[str, float]] = {}

        spherical_context: Optional[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = None

        for output_name in self.target_names:
            if output_name not in outputs:
                continue

            if output_name in self.scalar_target_names:
                atomic_predictions_by_block = {
                    key: torch.zeros(
                        1, dtype=edge_vectors.dtype, device=edge_vectors.device
                    )
                    for key in self.output_shapes[output_name].keys()
                }

                node_atomic_predictions_by_block = node_atomic_predictions_dict[
                    output_name
                ]
                edge_atomic_predictions_by_block = edge_atomic_predictions_dict[
                    output_name
                ]
                for i in range(len(node_atomic_predictions_by_block)):
                    node_atomic_prediction_block = node_atomic_predictions_by_block[i]
                    edge_atomic_prediction_block = edge_atomic_predictions_by_block[i]
                    for j, key in enumerate(atomic_predictions_by_block):
                        node_atomic_predictions = node_atomic_prediction_block[j]
                        edge_atomic_predictions = edge_atomic_prediction_block[j]
                        atomic_predictions_by_block[key] = atomic_predictions_by_block[
                            key
                        ] + (node_atomic_predictions + edge_atomic_predictions)

                if output_name == "non_conservative_stress":
                    block_key = list(atomic_predictions_by_block.keys())[0]
                    output_shapes_values = list(
                        self.output_shapes[output_name].values()
                    )
                    num_properties = output_shapes_values[0][-1]
                    atomic_predictions_by_block[block_key] = (
                        process_non_conservative_stress(
                            atomic_predictions_by_block[block_key],
                            systems,
                            system_indices,
                            num_properties,
                        )
                    )

                blocks = [
                    TensorBlock(
                        values=atomic_predictions_by_block[key].reshape([-1] + shape),
                        samples=sample_labels,
                        components=components,
                        properties=properties,
                    )
                    for key, shape, components, properties in zip(
                        self.output_shapes[output_name].keys(),
                        self.output_shapes[output_name].values(),
                        self.component_labels[output_name],
                        self.property_labels[output_name],
                        strict=True,
                    )
                ]
                atomic_predictions_tmap_dict[output_name] = TensorMap(
                    keys=self.key_labels[output_name],
                    blocks=blocks,
                )
                continue

            if spherical_context is None:
                spherical_context = self._compute_interatomic_vectors(systems)
            interatomic_vectors, centers, neighbors, species, _ = spherical_context
            basis_species = self._tensor_basis_species(species)

            blocks = torch.jit.annotate(List[TensorBlock], [])
            coefficient_norms: List[torch.Tensor] = []
            basis_norms: List[torch.Tensor] = []
            gram_traces: List[torch.Tensor] = []
            block_basis_gram_penalties: List[torch.Tensor] = []

            assert self.basis_calculators is not None
            found_basis_calculators = False
            for candidate_output_name, candidate_basis_calculators in (
                self.basis_calculators.items()
            ):
                if candidate_output_name == output_name:
                    found_basis_calculators = True
                    for block_index, (dict_key, tensor_basis_calculator) in enumerate(
                        candidate_basis_calculators.items()
                    ):
                        accumulated = (
                            node_atomic_predictions_dict[output_name][0][block_index]
                            + edge_atomic_predictions_dict[output_name][0][block_index]
                        )
                        for readout_index in range(
                            1, len(node_atomic_predictions_dict[output_name])
                        ):
                            node_atomic_predictions = node_atomic_predictions_dict[
                                output_name
                            ][readout_index][block_index]
                            edge_atomic_predictions = edge_atomic_predictions_dict[
                                output_name
                            ][readout_index][block_index]
                            accumulated = (
                                accumulated
                                + node_atomic_predictions
                                + edge_atomic_predictions
                            )
                        num_properties, basis_size = self.coefficient_shapes[
                            output_name
                        ][dict_key]
                        invariant_coefficients = accumulated.reshape(
                            accumulated.shape[0], num_properties, basis_size
                        )

                        tensor_basis = tensor_basis_calculator(
                            interatomic_vectors,
                            centers,
                            neighbors,
                            basis_species,
                            sample_labels.values,
                            None,
                        )

                        atomic_property_tensor = torch.einsum(
                            "spb, scb -> scp",
                            invariant_coefficients,
                            tensor_basis,
                        )
                        if len(self.component_labels[output_name][block_index]) == 0:
                            atomic_property_tensor = atomic_property_tensor.squeeze(1)

                        blocks.append(
                            TensorBlock(
                                values=atomic_property_tensor,
                                samples=sample_labels,
                                components=self.component_labels[output_name][
                                    block_index
                                ],
                                properties=self.property_labels[output_name][
                                    block_index
                                ],
                            )
                        )

                        coefficient_norms.append(
                            invariant_coefficients.pow(2).mean(dim=(1, 2)).mean()
                        )
                        basis_norms.append(
                            tensor_basis.pow(2).mean(dim=(1, 2)).mean()
                        )
                        gram = torch.einsum("scb,scd->sbd", tensor_basis, tensor_basis)
                        gram_traces.append(
                            torch.diagonal(gram, dim1=-2, dim2=-1).sum(dim=-1).mean()
                        )
                        block_basis_gram_penalties.append(
                            _basis_gram_chunk_penalty(tensor_basis)
                        )

            if not found_basis_calculators:
                raise RuntimeError(
                    f"Missing tensor basis calculators for spherical target '{output_name}'."
                )

            atomic_predictions_tmap_dict[output_name] = TensorMap(
                keys=self.key_labels[output_name], blocks=blocks
            )
            coefficient_penalties.extend(coefficient_norms)
            basis_gram_penalties.extend(block_basis_gram_penalties)
            diagnostics[output_name] = {
                "coefficient_norm_mean": float(
                    torch.stack(coefficient_norms).mean().detach()
                ),
                "basis_norm_mean": float(torch.stack(basis_norms).mean().detach()),
                "gram_trace_mean": float(torch.stack(gram_traces).mean().detach()),
                "basis_gram_penalty_mean": float(
                    torch.stack(block_basis_gram_penalties).mean().detach()
                ),
            }

        if coefficient_penalties:
            self._last_spherical_coefficient_penalty = torch.stack(
                coefficient_penalties
            ).mean()
        else:
            self._last_spherical_coefficient_penalty = torch.tensor(
                0.0, device=edge_vectors.device, dtype=edge_vectors.dtype
            )
        if basis_gram_penalties:
            self._last_basis_gram_penalty = torch.stack(basis_gram_penalties).mean()
        else:
            self._last_basis_gram_penalty = torch.tensor(
                0.0, device=edge_vectors.device, dtype=edge_vectors.dtype
            )
        self.latest_diagnostics = diagnostics

        if selected_atoms is not None:
            for output_name, tmap in atomic_predictions_tmap_dict.items():
                atomic_predictions_tmap_dict[output_name] = mts.slice(
                    tmap, axis="samples", selection=selected_atoms
                )

        for output_name in self.volume_normalized_target_names:
            if output_name in atomic_predictions_tmap_dict:
                atomic_predictions_tmap_dict[output_name] = normalize_by_volume(
                    atomic_predictions_tmap_dict[output_name], systems
                )

        for output_name, atomic_property in list(atomic_predictions_tmap_dict.items()):
            if outputs[output_name].per_atom:
                atomic_predictions_tmap_dict[output_name] = atomic_property
            else:
                atomic_predictions_tmap_dict[output_name] = sum_over_atoms(
                    atomic_property
                )

        return atomic_predictions_tmap_dict

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint: Dict[str, Any],
        context: Literal["restart", "finetune", "export"],
    ) -> "EPET":
        if context == "restart":
            logging.info(f"Using latest model from epoch {checkpoint['epoch']}")
            model_state_dict = checkpoint["model_state_dict"]
        elif context in {"finetune", "export"}:
            logging.info(f"Using best model from epoch {checkpoint['best_epoch']}")
            model_state_dict = checkpoint["best_model_state_dict"]
        else:
            raise ValueError("Unknown context tag for checkpoint loading!")

        model_data = checkpoint["model_data"]
        model = cls(
            hypers=model_data["model_hypers"],
            dataset_info=model_data["dataset_info"],
        )
        model_state_dict = dict(model_state_dict)
        model.finetune_config = model_state_dict.pop("finetune_config", {})
        try:
            dtype = next(
                value.dtype
                for value in model_state_dict.values()
                if isinstance(value, torch.Tensor) and value.is_floating_point()
            )
        except StopIteration as exc:
            raise RuntimeError(
                "Could not infer a floating dtype from the e-pet checkpoint."
            ) from exc

        model.to(dtype).load_state_dict(model_state_dict)
        model.additive_models[0].sync_tensor_maps()
        model.scaler.sync_tensor_maps()
        model.metadata = checkpoint["metadata"]
        return model

    @classmethod
    def upgrade_checkpoint(cls, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        if checkpoint["model_ckpt_version"] != cls.__checkpoint_version__:
            raise RuntimeError(
                "Unable to upgrade the checkpoint: the checkpoint is using model "
                f"version {checkpoint['model_ckpt_version']}, while the current "
                f"model version is {cls.__checkpoint_version__}."
            )
        return checkpoint

    def get_checkpoint(self) -> Dict[str, Any]:
        model_state_dict = self.state_dict()
        model_state_dict["finetune_config"] = self.finetune_config
        return {
            "architecture_name": "experimental.e_pet",
            "model_ckpt_version": self.__checkpoint_version__,
            "metadata": self.metadata,
            "model_data": {
                "model_hypers": self.e_pet_hypers,
                "dataset_info": self.dataset_info,
            },
            "epoch": None,
            "best_epoch": None,
            "model_state_dict": model_state_dict,
            "best_model_state_dict": self.state_dict(),
        }
