import random
from typing import Dict, List, Optional, Tuple

import metatensor.torch as mts
import torch
from metatensor.torch import TensorBlock, TensorMap
from metatomic.torch import NeighborListOptions, System, register_autograd_neighbors

from metatrain.utils import torch_jit_script_unless_coverage
from metatrain.utils.data import TargetInfo


def systems_to_batch(
    systems: List[System], nl_options: NeighborListOptions
) -> Dict[str, torch.Tensor]:
    """
    Convert a list of System objects directly to a GNN-batch-like dictionary.

    This function creates a torch-compile-friendly batch representation with
    stacked positions, cells, number of atoms per structure, atomic types,
    and center/neighbor indices.

    :param systems: List of System objects to batch
    :param nl_options: Neighbor list options to extract neighbor information
    :return: Dictionary containing batched tensors:
        - positions: stacked positions of all atoms [N_total, 3]
        - cells: stacked unit cells [N_structures, 3, 3]
        - species: atomic types of all atoms [N_total]
        - n_atoms: number of atoms per structure [N_structures]
        - cell_shifts: cell shift vectors for all pairs [N_pairs, 3]
        - centers: local atom indices within each structure [N_total]
        - center_indices: global center indices for all pairs [N_pairs]
        - neighbor_indices: global neighbor indices for all pairs [N_pairs]
        - structure_centers: structure index for each atom [N_total]
        - structure_pairs: structure index for each pair [N_pairs]
        - structure_offsets: cumulative atom offsets per structure [N_structures]
    """
    device = systems[0].positions.device

    positions_list = []
    species_list = []
    cells_list = []
    n_atoms_list: List[int] = []
    edge_index_list = []
    cell_shifts_list = []
    centers_list = []
    structures_centers_list = []
    structure_pairs_list = []

    cumulative_atoms = 0
    for i, system in enumerate(systems):
        n_atoms_i = len(system.positions)
        n_atoms_list.append(n_atoms_i)

        positions_list.append(system.positions)
        species_list.append(system.types)
        cells_list.append(system.cell)

        nl = system.get_neighbor_list(nl_options)
        samples = nl.samples.values
        edge_indices = samples[:, :2]  # local center/neighbor indices
        cell_shifts_item = samples[:, 2:]

        # Create global indices by adding cumulative offset
        global_center_indices = edge_indices[:, 0] + cumulative_atoms
        global_neighbor_indices = edge_indices[:, 1] + cumulative_atoms

        edge_index_list.append(
            torch.stack([global_center_indices, global_neighbor_indices], dim=1)
        )
        cell_shifts_list.append(cell_shifts_item)

        centers_list.append(torch.arange(n_atoms_i, device=device, dtype=torch.int32))
        structures_centers_list.append(
            torch.full((n_atoms_i,), i, device=device, dtype=torch.int32)
        )
        structure_pairs_list.append(
            torch.full((len(edge_indices),), i, device=device, dtype=torch.int32)
        )

        cumulative_atoms += n_atoms_i

    positions = torch.cat(positions_list, dim=0)
    species = torch.cat(species_list, dim=0)
    cells = torch.stack(cells_list, dim=0)
    n_atoms = torch.tensor(n_atoms_list, device=device, dtype=torch.int64)
    pairs = torch.cat(edge_index_list, dim=0)
    cell_shifts = torch.cat(cell_shifts_list, dim=0)
    centers = torch.cat(centers_list, dim=0)
    structure_centers = torch.cat(structures_centers_list, dim=0)
    structure_pairs = torch.cat(structure_pairs_list, dim=0)

    # Compute structure offsets (cumulative sum of n_atoms, starting with 0)
    structure_offsets = torch.zeros(len(systems), device=device, dtype=torch.int32)
    if len(systems) > 1:
        structure_offsets[1:] = torch.cumsum(n_atoms[:-1].to(torch.int32), dim=0)

    batch_dict = {
        "positions": positions,
        "cells": cells,
        "species": species,
        "n_atoms": n_atoms,
        "cell_shifts": cell_shifts,
        "centers": centers,
        "center_indices": pairs[:, 0],
        "neighbor_indices": pairs[:, 1],
        "structure_centers": structure_centers,
        "structure_pairs": structure_pairs,
        "structure_offsets": structure_offsets,
    }

    return batch_dict


def get_random_inversion() -> int:
    """
    Randomly choose an inversion factor (-1 or 1).

    :return: either -1 or 1
    """
    return random.choice([1, -1])


class InversionAugmenter:
    """
    A class to apply random inversions to a set of systems and their targets.

    :param target_info_dict: A dictionary mapping target names to their corresponding
        :class:`TargetInfo` objects. This is used to determine the type of targets and
        how to apply the augmentations.
    :param extra_data_info_dict: An optional dictionary mapping extra data names to
        their corresponding :py:class:`TargetInfo` objects. This is used to determine
        the type of extra data and how to apply the augmentations.
    """

    def __init__(
        self,
        target_info_dict: Dict[str, TargetInfo],
        extra_data_info_dict: Optional[Dict[str, TargetInfo]] = None,
    ):
        # checks on targets
        for target_info in target_info_dict.values():
            if target_info.is_cartesian:
                if len(target_info.layout.block(0).components) > 2:
                    raise ValueError(
                        "InversionAugmenter only supports Cartesian targets "
                        "with `rank<=2`."
                    )

        self.target_info_dict = target_info_dict
        if extra_data_info_dict is None:
            extra_data_info_dict = {}
        self.extra_data_info_dict = extra_data_info_dict

    def apply_random_augmentations(
        self,
        systems: List[System],
        targets: Dict[str, TensorMap],
        extra_data: Optional[Dict[str, TensorMap]] = None,
    ) -> Tuple[List[System], Dict[str, TensorMap], Dict[str, TensorMap]]:
        """
        Applies random augmentations to a number of ``System`` objects, their targets,
        and optionally extra data.

        :param systems: A list of :py:class:`System` objects to be augmented.
        :param targets: A dictionary mapping target names to their corresponding
            :py:class:`TensorMap` objects. These are the targets to be augmented.
        :param extra_data: An optional dictionary mapping extra data names to their
            corresponding :class:`TensorMap` objects. This extra data will also be
            augmented if provided.

        :return: A tuple containing the augmented systems and targets.
        """
        inversions = [get_random_inversion() for _ in range(len(systems))]
        return self.apply_augmentations(
            systems, targets, inversions, extra_data=extra_data
        )

    def apply_augmentations(
        self,
        systems: List[System],
        targets: Dict[str, TensorMap],
        inversions: List[int],
        extra_data: Optional[Dict[str, TensorMap]] = None,
    ) -> Tuple[List[System], Dict[str, TensorMap], Dict[str, TensorMap]]:
        """
        Applies augmentations to a number of ``System`` objects, their targets, and
        optionally extra data. The augmentations are defined by a list of rotations
        and a list of inversions.

        :param systems: A list of :py:class:`System` objects to be augmented.
        :param targets: A dictionary mapping target names to their corresponding
            :py:class:`TensorMap` objects. These are the targets to be augmented.
        :param rotations: A list of :class:`scipy.spatial.transform.Rotation` objects
            representing the rotations to be applied to each system.
        :param inversions: A list of integers (1 or -1) representing the
            inversion factors to be applied to each system.
        :param extra_data: An optional dictionary mapping extra data names to their
            corresponding :class:`TensorMap` objects. This extra data will also be
            augmented if provided.

        :return: A tuple containing the augmented systems and targets.
        """
        self._validate(systems, inversions)

        return _apply_augmentations(systems, targets, inversions, extra_data=extra_data)

    def _validate(self, systems: List[System], inversions: List[int]) -> None:
        if len(inversions) != len(systems):
            raise ValueError(
                "The number of inversions must match the number of systems."
            )
        if any(i not in [1, -1] for i in inversions):
            raise ValueError("Inversions must be either 1 or -1.")


def _apply_inversions_to_spherical_tensor_map(
    systems: List[System],
    target_tmap: TensorMap,
    inversions: List[int],
) -> TensorMap:
    new_blocks: List[TensorBlock] = []
    for key, block in target_tmap.items():
        ell, sigma = int(key[0]), int(key[1])
        values = block.values
        if "atom" in block.samples.names:
            split_values = torch.split(
                values, [len(system.positions) for system in systems]
            )
        else:
            split_values = torch.split(values, [1 for _ in systems])
        new_values = []
        ell = (len(block.components[0]) - 1) // 2
        for v, i in zip(split_values, inversions, strict=True):
            is_inverted = i == -1
            new_v = v.clone()
            if is_inverted:  # inversion
                new_v = new_v * (-1) ** ell * sigma
            new_values.append(new_v)
        new_values = torch.concatenate(new_values)
        new_block = TensorBlock(
            values=new_values,
            samples=block.samples,
            components=block.components,
            properties=block.properties,
        )
        new_blocks.append(new_block)

    return TensorMap(
        keys=target_tmap.keys,
        blocks=new_blocks,
    )


@torch_jit_script_unless_coverage  # script for speed
def _apply_augmentations(
    systems: List[System],
    targets: Dict[str, TensorMap],
    inversions: List[int],
    extra_data: Optional[Dict[str, TensorMap]] = None,
) -> Tuple[List[System], Dict[str, TensorMap], Dict[str, TensorMap]]:
    # Apply the transformations to the systems

    new_systems: List[System] = []
    for system, i in zip(systems, inversions, strict=True):
        new_system = System(
            positions=system.positions * i,
            types=system.types,
            cell=system.cell * i,
            pbc=system.pbc,
        )
        for data_name in system.known_data():
            data = system.get_data(data_name)
            # check if this data is easy to handle (scalar/vector), otherwise error out
            if len(data) != 1:
                raise ValueError(
                    f"System data '{data_name}' has {len(data)} blocks, which is not "
                    "supported by RotationalAugmenter. Only scalar and vector data are "
                    "supported."
                )
            if len(data.block().components) == 0:
                # scalar data, no change
                new_system.add_data(data_name, data)
            elif len(data.block().components) == 1 and data.block().components[
                0
            ].names == ["xyz"]:
                # this assumes that this is a proper vector (quite safe)
                new_system.add_data(
                    data_name,
                    TensorMap(
                        keys=data.keys,
                        blocks=[
                            TensorBlock(
                                values=data.block().values * i,
                                samples=data.block().samples,
                                components=data.block().components,
                                properties=data.block().properties,
                            )
                        ],
                    ),
                )
            else:
                raise ValueError(
                    f"System data '{data_name}' has components "
                    f"{data.block().components}, which are not supported by "
                    "InversionAugmenter. Only scalar and vector data are supported."
                )
        for options in system.known_neighbor_lists():
            neighbors = mts.detach_block(system.get_neighbor_list(options))

            neighbors.values[:] = (neighbors.values.squeeze(-1) * i).unsqueeze(-1)

            register_autograd_neighbors(system, neighbors)
            new_system.add_neighbor_list(options, neighbors)
        new_systems.append(new_system)

    # Apply the transformation to the targets and extra data
    new_targets: Dict[str, TensorMap] = {}
    new_extra_data: Dict[str, TensorMap] = {}

    # Do not transform any masks present in extra_data
    if extra_data is not None:
        mask_keys: List[str] = []
        for key in extra_data.keys():
            if key.endswith("_mask"):
                mask_keys.append(key)
        for key in mask_keys:
            new_extra_data[key] = extra_data.pop(key)

    for tensormap_dict, new_dict in zip(
        [targets, extra_data], [new_targets, new_extra_data], strict=True
    ):
        if tensormap_dict is None:
            continue
        assert tensormap_dict is not None
        for name, original_tmap in tensormap_dict.items():
            is_scalar = False
            if len(original_tmap.blocks()) == 1:
                if len(original_tmap.block().components) == 0:
                    is_scalar = True

            is_cartesian = False
            if len(original_tmap.blocks()) == 1:
                if len(original_tmap.block().components) > 0:
                    if "xyz" in original_tmap.block().components[0].names[0]:
                        is_cartesian = True

            is_spherical = all(
                len(block.components) == 1 and block.components[0].names == ["o3_mu"]
                for block in original_tmap.blocks()
            )

            if is_scalar:
                # no change for energies
                energy_block = TensorBlock(
                    values=original_tmap.block().values,
                    samples=original_tmap.block().samples,
                    components=original_tmap.block().components,
                    properties=original_tmap.block().properties,
                )
                if original_tmap.block().has_gradient("positions"):
                    # transform position gradients:
                    block = original_tmap.block().gradient("positions")
                    position_gradients = block.values.squeeze(-1)
                    split_sizes_forces = [
                        system.positions.shape[0] for system in systems
                    ]
                    split_position_gradients = torch.split(
                        position_gradients, split_sizes_forces
                    )
                    position_gradients = torch.cat(
                        [
                            split_position_gradients[i] * inversions[i]
                            for i in range(len(systems))
                        ]
                    )
                    energy_block.add_gradient(
                        "positions",
                        TensorBlock(
                            values=position_gradients.unsqueeze(-1),
                            samples=block.samples,
                            components=block.components,
                            properties=block.properties,
                        ),
                    )
                if original_tmap.block().has_gradient("strain"):
                    # transform strain gradients (rank-2 tensor), unchanged:
                    block = original_tmap.block().gradient("strain")
                    energy_block.add_gradient(
                        "strain",
                        TensorBlock(
                            values=block.values,
                            samples=block.samples,
                            components=block.components,
                            properties=block.properties,
                        ),
                    )
                new_dict[name] = TensorMap(
                    keys=original_tmap.keys,
                    blocks=[energy_block],
                )

            elif is_spherical:
                new_dict[name] = _apply_inversions_to_spherical_tensor_map(
                    systems, original_tmap, inversions
                )

            elif is_cartesian:
                rank = len(original_tmap.block().components)
                if rank == 1:
                    # transform Cartesian vector (assume proper, quite safe)
                    block = original_tmap.block()
                    vectors = block.values
                    if "atom" in original_tmap.block().samples.names:
                        split_vectors = torch.split(
                            vectors, [len(system.positions) for system in systems]
                        )
                    else:
                        split_vectors = torch.split(vectors, [1 for _ in systems])
                    new_vectors = []
                    for v, i in zip(split_vectors, inversions, strict=True):
                        new_v = v * i
                        new_vectors.append(new_v)
                    new_vectors = torch.cat(new_vectors)
                    new_dict[name] = TensorMap(
                        keys=original_tmap.keys,
                        blocks=[
                            TensorBlock(
                                values=new_vectors,
                                samples=block.samples,
                                components=block.components,
                                properties=block.properties,
                            )
                        ],
                    )
                elif rank == 2:
                    # assume proper tensor (quite safe), unchanged
                    new_dict[name] = original_tmap

    return new_systems, new_targets, new_extra_data
