from collections import namedtuple

import jax.numpy as jnp


JAXBatch = namedtuple(
    "JAXBatch",
    "positions, cells, numbers, centers, neighbors, "
    "cell_shifts, n_nodes, energies, forces",
)


def jax_structures_to_batch(structures):
    """Converts a list of JAX structures to a JAX batch.

    :param structures: A list of JAX structures.

    :return: A JAX batch.
    """
    n_nodes = jnp.array([len(structure.positions) for structure in structures])

    # concatenate after shifting
    shifted_centers = []
    shifted_neighbors = []
    shift = 0
    for structure in structures:
        shifted_centers.append(structure.centers + shift)
        shifted_neighbors.append(structure.neighbors + shift)
        shift += len(structure.positions)
    centers = jnp.concatenate(shifted_centers)
    neighbors = jnp.concatenate(shifted_neighbors)

    return JAXBatch(
        positions=jnp.concatenate([structure.positions for structure in structures]),
        cells=jnp.stack([structure.cell for structure in structures]),
        numbers=jnp.concatenate([structure.numbers for structure in structures]),
        centers=centers,
        neighbors=neighbors,
        cell_shifts=jnp.concatenate(
            [structure.cell_shifts for structure in structures]
        ),
        n_nodes=n_nodes,
        energies=jnp.stack([structure.energy for structure in structures]),
        forces=jnp.concatenate([structure.forces for structure in structures]),
    )


def calculate_padding_sizes(batch: JAXBatch):
    """Calculate the padding sizes for a batch. Works in powers of two.

    :param structures: A batch of structures.

    :return: A tuple with the padding sizes: nodes, edges, edges per node.
    """
    n_nodes = batch.positions.shape[0]
    n_edges = batch.neighbors.shape[0]
    n_edges_per_node = jnp.bincount(batch.neighbors).max()
    return (
        2 ** int(jnp.ceil(jnp.log2(n_nodes + 1))),
        2 ** int(jnp.ceil(jnp.log2(n_edges))),
        2 ** int(jnp.ceil(jnp.log2(n_edges_per_node))),
    )


def pad_batch(batch: JAXBatch, n_nodes: int, n_edges: int):
    """Pad a batch to the given sizes.

    :param batch: The batch to pad.
    :param n_nodes: The number of nodes to pad to.
    :param n_edges: The number of edges to pad to.

    :return: The padded batch.
    """

    # note: for node arrays, n_nodes - 1 is always
    # a padding value (see calculate_padding_sizes above)

    return JAXBatch(
        positions=jnp.pad(
            batch.positions, ((0, n_nodes - len(batch.positions)), (0, 0))
        ),
        cells=jnp.pad(batch.cells, ((0, n_nodes - len(batch.cells)), (0, 0), (0, 0))),
        numbers=jnp.pad(batch.numbers, (0, n_nodes - len(batch.numbers))),
        centers=jnp.pad(
            batch.centers,
            (0, n_edges - len(batch.centers)),
            mode="constant",
            constant_values=n_nodes - 1,
        ),
        neighbors=jnp.pad(
            batch.neighbors,
            (0, n_edges - len(batch.neighbors)),
            mode="constant",
            constant_values=n_nodes - 1,
        ),
        cell_shifts=jnp.pad(
            batch.cell_shifts, ((0, n_edges - len(batch.cell_shifts)), (0, 0))
        ),
        n_nodes=batch.n_nodes,
        energies=jnp.pad(batch.energies, (0, len(batch.n_nodes) - len(batch.energies))),
        forces=jnp.pad(batch.forces, (0, n_edges - len(batch.forces))),
    )
