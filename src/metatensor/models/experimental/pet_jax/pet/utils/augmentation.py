import random

import numpy as np
from scipy.spatial.transform import Rotation

from .mts_to_structure import Structure


def apply_random_augmentation(structure: Structure):
    """
    Apply a random augmentation to a ``Structure``.

    :param structure: The structure to augment.

    :return: The augmented structure.
    """

    transformation = get_random_augmentation()
    return Structure(
        positions=structure.positions @ transformation.T,
        cell=structure.cell @ transformation.T,
        numbers=structure.numbers,
        centers=structure.centers,
        neighbors=structure.neighbors,
        cell_shifts=structure.cell_shifts,
        energy=structure.energy,
        forces=structure.forces @ transformation.T,
    )


def get_random_augmentation():

    transformation = Rotation.random().as_matrix()
    invert = random.choice([True, False])
    if invert:
        transformation *= -1
    return transformation
