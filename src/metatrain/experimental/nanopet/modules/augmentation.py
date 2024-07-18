import random

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.atomistic import System
from scipy.spatial.transform import Rotation
from typing import Dict, List, Tuple


def apply_random_augmentations(systems: List[System], targets: Dict[str, TensorMap]) -> Tuple[List[System], Dict[str, TensorMap]]:
    """
    Apply a random augmentation to a number of ``System`` objects and its targets.
    """

    transformations = [
        torch.tensor(
            get_random_augmentation(), dtype=systems[0].positions.dtype
        )
        for _ in range(len(systems))
    ]

    return _apply_random_augmentations(systems, targets, transformations)


@torch.jit.script
def _apply_random_augmentations(systems: List[System], targets: Dict[str, TensorMap], transformations: List[torch.Tensor]) -> Tuple[List[System], Dict[str, TensorMap]]:

    split_sizes_forces = [system.positions.shape[0] for system in systems]

    # Apply the transformations to the systems
    new_systems: List[System] = []
    for system, transformation in zip(systems, transformations):
        new_system = System(
            positions=system.positions @ transformation.T,
            types=system.types,
            cell=system.cell @ transformation.T,
        )
        for nl_options in system.known_neighbor_lists():
            old_nl = system.get_neighbor_list(nl_options)
            new_system.add_neighbor_list(
                nl_options,
                TensorBlock(
                    values=(old_nl.values.squeeze(-1) @ transformation.T).unsqueeze(-1),
                    samples=old_nl.samples,
                    components=old_nl.components,
                    properties=old_nl.properties,
                ),
            )
        new_systems.append(new_system)

    # Apply the transformation to the targets
    new_targets: Dict[str, TensorMap] = {}
    for name, target_tmap in targets.items():
        # no change for energies
        energy_block = TensorBlock(
            values=target_tmap.block().values,
            samples=target_tmap.block().samples,
            components=target_tmap.block().components,
            properties=target_tmap.block().properties
        )
        if target_tmap.block().has_gradient("positions"):
            # transform position gradients:
            block = target_tmap.block().gradient("positions")
            position_gradients = block.values.squeeze(-1)
            split_position_gradients = torch.split(position_gradients, split_sizes_forces)
            position_gradients = torch.cat(
                [split_position_gradients[i] @ transformations[i].T for i in range(len(systems))]
            )
            energy_block.add_gradient(
                "positions",
                TensorBlock(
                    values=position_gradients.unsqueeze(-1),
                    samples=block.samples,
                    components=block.components,
                    properties=block.properties
                )
            )
        if target_tmap.block().has_gradient("strain"):
            # transform strain gradients (rank 2 tensor):
            block = target_tmap.block().gradient("strain")
            strain_gradients = block.values.squeeze(-1)
            split_strain_gradients = torch.split(strain_gradients, 1)
            new_strain_gradients = torch.stack(
                [transformations[i] @ split_strain_gradients[i].squeeze(0) @ transformations[i].T for i in range(len(systems))], dim=0
            )
            energy_block.add_gradient(
                "strain",
                TensorBlock(
                    values=new_strain_gradients.unsqueeze(-1),
                    samples=block.samples,
                    components=block.components,
                    properties=block.properties
                )
            )
        new_targets[name] = TensorMap(
            keys=target_tmap.keys,
            blocks=[energy_block],
        )

    return new_systems, new_targets


def get_random_augmentation():

    transformation = Rotation.random().as_matrix()
    invert = random.choice([True, False])
    if invert:
        transformation *= -1
    return transformation
