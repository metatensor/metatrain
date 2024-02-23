from collections import namedtuple

import jax.numpy as jnp
import numpy as np

from .mts_to_structure import Structure


JAXStructure = namedtuple(
    "JAXStructure",
    "positions, cell, numbers, centers, neighbors, cell_shifts, energy, forces",
)


def structure_to_jax(structure: Structure):
    """Converts a Structure to a JAX dictionary.

    :param structure: The structure to convert.

    :return: The same named tuple, but with jnp arrays.
    """

    if not np.all(structure.centers[1:] >= structure.centers[:-1]):
        raise ValueError(
            "centers array of the neighbor list is not sorted. "
            "This is required for the JAX implementation."
        )

    return JAXStructure(
        positions=jnp.array(structure.positions),
        cell=jnp.array(structure.cell),
        numbers=jnp.array(structure.numbers),
        centers=jnp.array(structure.centers),
        neighbors=jnp.array(structure.neighbors),
        cell_shifts=jnp.array(structure.cell_shifts),
        energy=jnp.array(structure.energy),
        forces=jnp.array(structure.forces),
    )
