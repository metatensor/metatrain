import random

import torch
from metatensor.torch import TensorBlock
from metatensor.torch.atomistic import System
from scipy.spatial.transform import Rotation


def apply_random_augmentation(system: System) -> System:
    """
    Apply a random augmentation to a ``System``.

    :param structure: The structure to augment.
    :return: The augmented structure.
    """

    transformation = torch.tensor(
        get_random_augmentation(), dtype=system.positions.dtype
    )
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

    return new_system


def get_random_augmentation():

    transformation = Rotation.random().as_matrix()
    invert = random.choice([True, False])
    if invert:
        transformation *= -1
    return transformation
