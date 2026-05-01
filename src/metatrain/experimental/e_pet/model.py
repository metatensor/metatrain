import copy
import logging
import math
from typing import Any, Dict, List, Literal, Optional, Tuple

import metatensor.torch as mts
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import ModelMetadata, ModelOutput, System
from torch.nn import Linear, ModuleDict, ModuleList, Sequential, SiLU

from metatrain.pet.model import (
    PET,
    _irrep_key,
    _parse_shared_selector,
    _shared_selector,
    _validate_shared_head_groups,
    get_last_layer_features_name,
    normalize_by_volume,
    process_non_conservative_stress,
)
from metatrain.pet.modules.structures import concatenate_structures
from metatrain.soap_bpnn.modules.tensor_basis import TensorBasis
from metatrain.utils.data import DatasetInfo, TargetInfo
from metatrain.utils.data.atomic_basis_helpers import densify_atomic_basis_target
from metatrain.utils.data.target_info import get_generic_target_info
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


def _whiten_tensor_basis(
    tensor_basis: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Canonicalize basis coordinates with a regularized Gram inverse square root."""
    basis_width = tensor_basis.shape[2]
    if basis_width == 0:
        return tensor_basis

    # This is equivalent to ``B @ (B^T B + eps I)^(-1/2)`` on the non-null
    # subspace, but is more stable than explicitly diagonalizing ``B^T B`` in
    # float32 when the raw basis is ill-conditioned or overcomplete.
    left_vectors, singular_values, right_vectors_t = torch.linalg.svd(
        tensor_basis,
        full_matrices=False,
    )
    scaled_singular_values = singular_values / torch.sqrt(
        singular_values.pow(2) + epsilon
    )
    return torch.einsum(
        "sci,si,sib->scb",
        left_vectors,
        scaled_singular_values,
        right_vectors_t,
    )


def _extra_l1_vector_basis_branches(
    tensor_basis_hypers: Dict[str, Any]
) -> list[dict[str, Any]]:
    return copy.deepcopy(
        tensor_basis_hypers.get("extra_l1_vector_basis_branches", [])
    )


# Cartesian rank-2 targets stay public Cartesian targets for losses/metrics. E-PET
# only uses the hardcoded l=0/l=2 representation internally to drive tensor-basis
# readouts, then reconstructs the symmetric Cartesian tensor before post-processing.
def _is_cartesian_rank2_target(target_info: TargetInfo) -> bool:
    if not target_info.is_cartesian:
        return False
    components = target_info.layout.block().components
    return len(components) == 2 and all(len(component) == 3 for component in components)


def _cartesian_rank2_spherical_target_info(
    target_name: str, target_info: TargetInfo
) -> TargetInfo:
    return get_generic_target_info(
        target_name,
        {
            "quantity": target_info.quantity,
            "unit": target_info.unit,
            "description": target_info.description,
            "type": {
                "spherical": {
                    "irreps": [
                        {"o3_lambda": 0, "o3_sigma": 1},
                        {"o3_lambda": 2, "o3_sigma": 1},
                    ],
                }
            },
            "num_subtargets": len(target_info.layout.block().properties),
            "per_atom": target_info.per_atom,
        },
    )


def _cartesian_rank2_to_spherical_components(
    cartesian: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xx = cartesian[:, 0, 0]
    yy = cartesian[:, 1, 1]
    zz = cartesian[:, 2, 2]
    xy = cartesian[:, 0, 1]
    yz = cartesian[:, 1, 2]
    xz = cartesian[:, 0, 2]

    l0 = ((xx + yy + zz) / math.sqrt(3.0)).unsqueeze(1)
    l2 = torch.stack(
        [
            math.sqrt(2.0) * xy,
            math.sqrt(2.0) * yz,
            (2.0 * zz - xx - yy) / math.sqrt(6.0),
            math.sqrt(2.0) * xz,
            (xx - yy) / math.sqrt(2.0),
        ],
        dim=1,
    )
    return l0, l2


def _spherical_components_to_cartesian_rank2(
    spherical_tensor_map: TensorMap,
    public_layout: TensorMap,
) -> TensorMap:
    l0_block = spherical_tensor_map.block(0)
    l2_block = spherical_tensor_map.block(1)
    l0 = l0_block.values[:, 0]
    l2 = l2_block.values

    cartesian = torch.empty(
        l0.shape[0],
        3,
        3,
        l0.shape[1],
        dtype=l0.dtype,
        device=l0.device,
    )
    sqrt_2 = math.sqrt(2.0)
    sqrt_3 = math.sqrt(3.0)
    sqrt_6 = math.sqrt(6.0)

    cartesian[:, 0, 0] = l0 / sqrt_3 - l2[:, 2] / sqrt_6 + l2[:, 4] / sqrt_2
    cartesian[:, 1, 1] = l0 / sqrt_3 - l2[:, 2] / sqrt_6 - l2[:, 4] / sqrt_2
    cartesian[:, 2, 2] = l0 / sqrt_3 + 2.0 * l2[:, 2] / sqrt_6
    cartesian[:, 0, 1] = l2[:, 0] / sqrt_2
    cartesian[:, 1, 0] = cartesian[:, 0, 1]
    cartesian[:, 1, 2] = l2[:, 1] / sqrt_2
    cartesian[:, 2, 1] = cartesian[:, 1, 2]
    cartesian[:, 0, 2] = l2[:, 3] / sqrt_2
    cartesian[:, 2, 0] = cartesian[:, 0, 2]

    public_block = public_layout.block()
    block = TensorBlock(
        values=cartesian,
        samples=l0_block.samples,
        components=public_block.components,
        properties=public_block.properties,
    )
    return TensorMap(keys=public_layout.keys, blocks=[block])


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
    cartesian_rank2_target_names: List[str]
    cartesian_rank2_public_layouts: Dict[str, TensorMap]
    volume_normalized_target_names: List[str]
    tensor_basis_legacy: bool
    coefficient_shapes: Dict[str, Dict[str, Tuple[int, int]]]
    target_head_keys: Dict[str, List[str]]
    block_to_head_key: Dict[str, Dict[str, str]]
    block_irrep_keys: Dict[str, Dict[str, str]]
    shared_head_selectors: Dict[str, str]
    top_level_shared_head_selectors: Dict[str, str]

    def __init__(self, hypers: ModelHypers, dataset_info: DatasetInfo) -> None:
        self.e_pet_hypers = copy.deepcopy(hypers)
        self.tensor_basis_hypers = copy.deepcopy(hypers["tensor_basis_defaults"])
        self.tensor_basis_hypers.setdefault("extra_l1_vector_basis_branches", [])
        self.basis_normalization = str(
            self.tensor_basis_hypers.get("basis_normalization", "none")
        )
        if self.basis_normalization not in ("none", "whiten"):
            raise ValueError(
                "Unknown E-PET tensor-basis normalization "
                f"{self.basis_normalization!r}. Expected 'none' or 'whiten'."
            )
        self.basis_normalization_epsilon = float(
            self.tensor_basis_hypers.get("basis_normalization_epsilon", 1.0e-6)
        )
        if self.basis_normalization_epsilon <= 0.0:
            raise ValueError("basis_normalization_epsilon must be positive.")
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
        self.cartesian_rank2_target_names = []
        self.cartesian_rank2_public_layouts: Dict[str, TensorMap] = {}
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
        self._last_spherical_coefficient_penalty_without_l0 = torch.tensor(0.0)
        self._last_basis_gram_penalty = torch.tensor(0.0)
        super().__init__(copy.deepcopy(hypers["pet"]), dataset_info)
        self.volume_normalized_target_names = list(
            hypers.get("volume_normalized_targets", [])
        )
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

    def _validate_and_build_shared_head_groups(
        self, dataset_info: DatasetInfo
    ) -> Dict[str, str]:
        for selectors in self.shared_head_groups_config.values():
            for selector in selectors:
                target_name, _ = _parse_shared_selector(selector)
                if (
                    target_name in dataset_info.targets
                    and dataset_info.targets[target_name].is_atomic_basis
                ):
                    raise ValueError(
                        "Atomic-basis targets cannot appear in "
                        f"shared_head_groups: {target_name!r}."
                    )

        return _validate_shared_head_groups(
            self.shared_head_groups_config,
            dataset_info,
            "Only scalar and spherical targets can appear in shared_head_groups; "
            "target '{}' is unsupported.",
            irrep_head_groups_config=self.irrep_head_groups_config,
            allow_same_group_duplicates=True,
        )

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

        atomic_basis_targets = sorted(
            target_name
            for target_name in self.irrep_head_groups_config
            if dataset_info.targets[target_name].is_atomic_basis
        )
        if atomic_basis_targets:
            raise ValueError(
                "Atomic-basis targets cannot appear in irrep_head_groups: "
                f"{atomic_basis_targets}."
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
        if target_info.is_scalar:
            self._add_scalar_output(target_name, target_info)
            return

        if target_info.is_spherical:
            self._add_spherical_output(target_name, target_info)
            return

        if _is_cartesian_rank2_target(target_info):
            self._add_cartesian_rank2_output(target_name, target_info)
            return

        if target_info.is_cartesian:
            raise ValueError(
                "experimental.e_pet supports Cartesian direct targets only for "
                "rank-2 tensors in v1."
            )

        raise ValueError(
            "experimental.e_pet supports only scalar, spherical, and Cartesian "
            "rank-2 targets in v1."
        )

    def _add_scalar_output(self, target_name: str, target_info: TargetInfo) -> None:
        self.scalar_target_names.append(target_name)
        super()._add_output(target_name, target_info)
        head_key = self.top_level_shared_head_selectors.get(
            _shared_selector(target_name), target_name
        )
        self.target_head_keys[target_name] = [head_key]
        if head_key == target_name:
            return

        if head_key in self.node_heads:
            del self.node_heads[target_name]
            del self.edge_heads[target_name]
            return

        self.node_heads[head_key] = self.node_heads[target_name]
        del self.node_heads[target_name]
        self.edge_heads[head_key] = self.edge_heads[target_name]
        del self.edge_heads[target_name]

    def _basis_size(
        self,
        key: Labels,
        num_components: int,
        extra_l1_vector_basis_branches: list[dict[str, Any]],
    ) -> int:
        if (
            num_components == 3
            and int(key[0]) == 1
            and int(key[1]) == 1
            and len(extra_l1_vector_basis_branches) > 0
        ):
            return (1 + len(extra_l1_vector_basis_branches)) * num_components
        if num_components in (1, 3):
            return num_components
        if self.tensor_basis_hypers["add_lambda_basis"]:
            return 2 * num_components
        return num_components

    def _add_spherical_output(
        self, target_name: str, target_info: TargetInfo
    ) -> None:
        if target_info.is_atomic_basis and not target_info.per_atom:
            raise ValueError(
                "experimental.e_pet currently supports only per-atom spherical "
                f"atomic-basis targets; target '{target_name}' is per-structure."
            )

        extra_l1_vector_basis_branches = _extra_l1_vector_basis_branches(
            self.tensor_basis_hypers
        )

        self.spherical_target_names.append(target_name)
        if self.basis_calculators is None:
            self.basis_calculators = ModuleDict({})

        output_layout = target_info.layout
        if target_info.is_atomic_basis:
            output_layout = densify_atomic_basis_target(output_layout, output_layout)
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
            irrep_key = _irrep_key(key)
            seen_irrep_keys.add(irrep_key)
            self.block_irrep_keys[target_name][dict_key] = irrep_key

            shared_selector = _shared_selector(target_name, irrep_key)
            if target_info.is_atomic_basis:
                head_key = target_name
            elif shared_selector in self.top_level_shared_head_selectors:
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
            basis_size = self._basis_size(
                key, num_components, extra_l1_vector_basis_branches
            )

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

    def _add_cartesian_rank2_output(
        self, target_name: str, target_info: TargetInfo
    ) -> None:
        self.cartesian_rank2_target_names.append(target_name)
        self.cartesian_rank2_public_layouts[target_name] = target_info.layout
        self._add_spherical_output(
            target_name,
            _cartesian_rank2_spherical_target_info(target_name, target_info),
        )

    def get_regularization_loss(
        self, exclude_spherical_l0: bool = False
    ) -> torch.Tensor:
        if exclude_spherical_l0:
            return self._last_spherical_coefficient_penalty_without_l0
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

    def _normalize_tensor_basis(self, tensor_basis: torch.Tensor) -> torch.Tensor:
        if self.basis_normalization == "none":
            return tensor_basis
        if self.basis_normalization == "whiten":
            return _whiten_tensor_basis(
                tensor_basis,
                self.basis_normalization_epsilon,
            )
        raise RuntimeError(
            f"Unsupported E-PET tensor-basis normalization {self.basis_normalization!r}."
        )

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
                node_atomic_predictions_dict[output_name] = torch.jit.annotate(
                    List[List[torch.Tensor]], []
                )
                for readout_index, node_last_layer in enumerate(node_last_layers):
                    block_predictions: List[torch.Tensor] = []
                    for block_key, node_last_layer_by_block in node_last_layer.items():
                        if output_name in self.scalar_target_names:
                            head_key = self.target_head_keys[output_name][0]
                        else:
                            head_key = self.block_to_head_key[output_name][block_key]
                        node_last_layer_features = node_last_layer_features_dict[
                            head_key
                        ][readout_index]
                        block_predictions.append(
                            node_last_layer_by_block(node_last_layer_features)
                        )
                    node_atomic_predictions_dict[output_name].append(block_predictions)

        for output_name, edge_last_layers in self.edge_last_layers.items():
            if output_name in outputs:
                edge_atomic_predictions_dict[output_name] = torch.jit.annotate(
                    List[List[torch.Tensor]], []
                )
                for readout_index, edge_last_layer in enumerate(edge_last_layers):
                    block_predictions: List[torch.Tensor] = []
                    for block_key, edge_last_layer_by_block in edge_last_layer.items():
                        if output_name in self.scalar_target_names:
                            head_key = self.target_head_keys[output_name][0]
                        else:
                            head_key = self.block_to_head_key[output_name][block_key]
                        edge_last_layer_features = edge_last_layer_features_dict[
                            head_key
                        ][readout_index]
                        edge_atomic_predictions = edge_last_layer_by_block(
                            edge_last_layer_features
                        )
                        expanded_padding_mask = padding_mask[..., None].repeat(
                            1, 1, edge_atomic_predictions.shape[2]
                        )
                        edge_atomic_predictions = torch.where(
                            ~expanded_padding_mask, 0.0, edge_atomic_predictions
                        )
                        block_predictions.append(
                            (
                                edge_atomic_predictions * cutoff_factors[:, :, None]
                            ).sum(dim=1)
                        )
                    edge_atomic_predictions_dict[output_name].append(block_predictions)

        return node_atomic_predictions_dict, edge_atomic_predictions_dict

    def _build_scalar_atomic_prediction(
        self,
        output_name: str,
        systems: List[System],
        node_atomic_predictions_dict: Dict[str, List[List[torch.Tensor]]],
        edge_atomic_predictions_dict: Dict[str, List[List[torch.Tensor]]],
        edge_vectors: torch.Tensor,
        system_indices: torch.Tensor,
        sample_labels: Labels,
    ) -> TensorMap:
        atomic_predictions_by_block = {
            key: torch.zeros(
                1, dtype=edge_vectors.dtype, device=edge_vectors.device
            )
            for key in self.output_shapes[output_name].keys()
        }

        node_atomic_predictions_by_block = node_atomic_predictions_dict[output_name]
        edge_atomic_predictions_by_block = edge_atomic_predictions_dict[output_name]
        for readout_index in range(len(node_atomic_predictions_by_block)):
            node_atomic_prediction_block = node_atomic_predictions_by_block[
                readout_index
            ]
            edge_atomic_prediction_block = edge_atomic_predictions_by_block[
                readout_index
            ]
            for block_index, key in enumerate(atomic_predictions_by_block):
                atomic_predictions_by_block[key] = atomic_predictions_by_block[
                    key
                ] + (
                    node_atomic_prediction_block[block_index]
                    + edge_atomic_prediction_block[block_index]
                )

        if output_name == "non_conservative_stress":
            block_key = list(atomic_predictions_by_block.keys())[0]
            output_shapes_values = list(self.output_shapes[output_name].values())
            num_properties = output_shapes_values[0][-1]
            atomic_predictions_by_block[block_key] = process_non_conservative_stress(
                atomic_predictions_by_block[block_key],
                systems,
                system_indices,
                num_properties,
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
        return TensorMap(keys=self.key_labels[output_name], blocks=blocks)

    def _accumulate_spherical_coefficients(
        self,
        output_name: str,
        block_index: int,
        node_atomic_predictions_dict: Dict[str, List[List[torch.Tensor]]],
        edge_atomic_predictions_dict: Dict[str, List[List[torch.Tensor]]],
    ) -> torch.Tensor:
        accumulated = (
            node_atomic_predictions_dict[output_name][0][block_index]
            + edge_atomic_predictions_dict[output_name][0][block_index]
        )
        for readout_index in range(1, len(node_atomic_predictions_dict[output_name])):
            accumulated = (
                accumulated
                + node_atomic_predictions_dict[output_name][readout_index][block_index]
                + edge_atomic_predictions_dict[output_name][readout_index][block_index]
            )
        return accumulated

    def _build_spherical_atomic_prediction(
        self,
        output_name: str,
        node_atomic_predictions_dict: Dict[str, List[List[torch.Tensor]]],
        edge_atomic_predictions_dict: Dict[str, List[List[torch.Tensor]]],
        sample_labels: Labels,
        interatomic_vectors: torch.Tensor,
        centers: torch.Tensor,
        neighbors: torch.Tensor,
        basis_species: torch.Tensor,
    ) -> Tuple[TensorMap, List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        blocks = torch.jit.annotate(List[TensorBlock], [])
        coefficient_norms: List[torch.Tensor] = []
        coefficient_norms_without_l0: List[torch.Tensor] = []
        block_basis_gram_penalties: List[torch.Tensor] = []

        if self.basis_calculators is None:
            raise RuntimeError(
                f"Missing tensor basis calculators for spherical target '{output_name}'."
            )

        for block_index, (dict_key, coefficient_shape) in enumerate(
            self.coefficient_shapes[output_name].items()
        ):
            accumulated = self._accumulate_spherical_coefficients(
                output_name,
                block_index,
                node_atomic_predictions_dict,
                edge_atomic_predictions_dict,
            )
            num_properties, basis_size = coefficient_shape
            invariant_coefficients = accumulated.reshape(
                accumulated.shape[0], num_properties, basis_size
            )

            tensor_basis = torch.empty(
                0, dtype=interatomic_vectors.dtype, device=interatomic_vectors.device
            )
            for output_name_basis, basis_calculators_by_block in (
                self.basis_calculators.items()
            ):
                # TorchScript can not compile direct nested ModuleDict lookup with
                # target names such as "mtt::..."; follow the PET/SOAP-BPNN export
                # pattern and select the module through static ModuleDict iteration.
                if output_name_basis == output_name:
                    for basis_calculator_key, basis_calculator in (
                        basis_calculators_by_block.items()
                    ):
                        if basis_calculator_key == dict_key:
                            tensor_basis = basis_calculator(
                                interatomic_vectors,
                                centers,
                                neighbors,
                                basis_species,
                                sample_labels.values,
                                None,
                            )
            if tensor_basis.numel() == 0:
                raise RuntimeError(
                    "Missing tensor basis calculator for spherical block "
                    f"'{output_name}:{dict_key}'."
                )

            tensor_basis = self._normalize_tensor_basis(tensor_basis)
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
                    components=self.component_labels[output_name][block_index],
                    properties=self.property_labels[output_name][block_index],
                )
            )

            coefficient_norm = invariant_coefficients.pow(2).mean(dim=(1, 2)).mean()
            coefficient_norms.append(coefficient_norm)
            if self.block_irrep_keys[output_name][dict_key] != "0,1":
                coefficient_norms_without_l0.append(coefficient_norm)
            block_basis_gram_penalties.append(_basis_gram_chunk_penalty(tensor_basis))

        return (
            TensorMap(keys=self.key_labels[output_name], blocks=blocks),
            coefficient_norms,
            coefficient_norms_without_l0,
            block_basis_gram_penalties,
        )

    def _build_cartesian_rank2_atomic_prediction(
        self,
        output_name: str,
        node_atomic_predictions_dict: Dict[str, List[List[torch.Tensor]]],
        edge_atomic_predictions_dict: Dict[str, List[List[torch.Tensor]]],
        sample_labels: Labels,
        interatomic_vectors: torch.Tensor,
        centers: torch.Tensor,
        neighbors: torch.Tensor,
        basis_species: torch.Tensor,
    ) -> Tuple[TensorMap, List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        (
            spherical_tensor_map,
            coefficient_norms,
            coefficient_norms_without_l0,
            basis_gram_penalties,
        ) = self._build_spherical_atomic_prediction(
            output_name,
            node_atomic_predictions_dict,
            edge_atomic_predictions_dict,
            sample_labels,
            interatomic_vectors,
            centers,
            neighbors,
            basis_species,
        )
        return (
            _spherical_components_to_cartesian_rank2(
                spherical_tensor_map,
                self.cartesian_rank2_public_layouts[output_name],
            ),
            coefficient_norms,
            coefficient_norms_without_l0,
            basis_gram_penalties,
        )

    def _set_last_regularization_losses(
        self,
        coefficient_penalties: List[torch.Tensor],
        coefficient_penalties_without_l0: List[torch.Tensor],
        basis_gram_penalties: List[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if coefficient_penalties:
            self._last_spherical_coefficient_penalty = torch.stack(
                coefficient_penalties
            ).mean()
        else:
            self._last_spherical_coefficient_penalty = torch.tensor(
                0.0, device=device, dtype=dtype
            )

        if coefficient_penalties_without_l0:
            self._last_spherical_coefficient_penalty_without_l0 = torch.stack(
                coefficient_penalties_without_l0
            ).mean()
        else:
            self._last_spherical_coefficient_penalty_without_l0 = torch.tensor(
                0.0, device=device, dtype=dtype
            )

        if basis_gram_penalties:
            self._last_basis_gram_penalty = torch.stack(basis_gram_penalties).mean()
        else:
            self._last_basis_gram_penalty = torch.tensor(
                0.0, device=device, dtype=dtype
            )

    def _postprocess_atomic_predictions(
        self,
        atomic_predictions_tmap_dict: Dict[str, TensorMap],
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
    ) -> Dict[str, TensorMap]:
        if selected_atoms is not None:
            for output_name, tmap in atomic_predictions_tmap_dict.items():
                atomic_predictions_tmap_dict[output_name] = mts.slice(
                    tmap, axis="samples", selection=selected_atoms
                )

        for output_name, atomic_property in list(atomic_predictions_tmap_dict.items()):
            if outputs[output_name].per_atom:
                atomic_predictions_tmap_dict[output_name] = atomic_property
            else:
                atomic_predictions_tmap_dict[output_name] = sum_over_atoms(
                    atomic_property
                )

        for output_name in self.volume_normalized_target_names:
            if (
                output_name in atomic_predictions_tmap_dict
                and not outputs[output_name].per_atom
                and output_name != "non_conservative_stress"
            ):
                atomic_predictions_tmap_dict[output_name] = normalize_by_volume(
                    atomic_predictions_tmap_dict[output_name], systems
                )

        return atomic_predictions_tmap_dict

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
        coefficient_penalties_without_l0: List[torch.Tensor] = []
        basis_gram_penalties: List[torch.Tensor] = []

        spherical_context: Optional[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = None

        for output_name in self.target_names:
            if output_name not in outputs:
                continue

            if output_name in self.scalar_target_names:
                atomic_predictions_tmap_dict[output_name] = (
                    self._build_scalar_atomic_prediction(
                        output_name,
                        systems,
                        node_atomic_predictions_dict,
                        edge_atomic_predictions_dict,
                        edge_vectors,
                        system_indices,
                        sample_labels,
                    )
                )
                continue

            if spherical_context is None:
                spherical_context = self._compute_interatomic_vectors(systems)
            interatomic_vectors, centers, neighbors, species, _ = spherical_context
            basis_species = self._tensor_basis_species(species)

            if output_name in self.cartesian_rank2_target_names:
                (
                    atomic_predictions_tmap_dict[output_name],
                    coefficient_norms,
                    coefficient_norms_without_l0,
                    block_basis_gram_penalties,
                ) = self._build_cartesian_rank2_atomic_prediction(
                    output_name,
                    node_atomic_predictions_dict,
                    edge_atomic_predictions_dict,
                    sample_labels,
                    interatomic_vectors,
                    centers,
                    neighbors,
                    basis_species,
                )
            else:
                (
                    atomic_predictions_tmap_dict[output_name],
                    coefficient_norms,
                    coefficient_norms_without_l0,
                    block_basis_gram_penalties,
                ) = self._build_spherical_atomic_prediction(
                    output_name,
                    node_atomic_predictions_dict,
                    edge_atomic_predictions_dict,
                    sample_labels,
                    interatomic_vectors,
                    centers,
                    neighbors,
                    basis_species,
                )
            coefficient_penalties.extend(coefficient_norms)
            coefficient_penalties_without_l0.extend(coefficient_norms_without_l0)
            basis_gram_penalties.extend(block_basis_gram_penalties)

        self._set_last_regularization_losses(
            coefficient_penalties,
            coefficient_penalties_without_l0,
            basis_gram_penalties,
            edge_vectors.device,
            edge_vectors.dtype,
        )

        return self._postprocess_atomic_predictions(
            atomic_predictions_tmap_dict,
            systems,
            outputs,
            selected_atoms,
        )

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
