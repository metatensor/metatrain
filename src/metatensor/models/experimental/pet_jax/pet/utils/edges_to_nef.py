import jax
import jax.numpy as jnp


def loop_body(i, value):
    array, t = value
    # jax.debug.print("{x}", x=jnp.nonzero(array == i, size=1, fill_value=-1)[0][0])
    t = t.at[i].set(jnp.nonzero(array == i, size=1)[0][0])
    return array, t


def get_first_occurrences(array, n_nodes: int):
    return jax.lax.fori_loop(
        0, n_nodes, loop_body, (array, jnp.empty((n_nodes,), dtype=jnp.int32))
    )[1]


def get_nef_indices(centers, n_nodes: int, n_edges_per_node: int):

    first_occurrences = get_first_occurrences(centers, n_nodes)
    an = jnp.repeat(
        jnp.arange(n_edges_per_node).reshape(1, n_edges_per_node), n_nodes, 0
    )
    mask = an < jnp.bincount(centers, length=n_nodes).reshape(n_nodes, 1)

    return (first_occurrences, an, mask)


def edge_array_to_nef(edge_array, nef_indices, fill_value):
    """Converts an edge array to a NEF array."""

    first_occurrences, an, mask = nef_indices

    index_array = first_occurrences.reshape(-1, 1) + an
    return jnp.where(
        mask.reshape(mask.shape + (1,) * (len(edge_array.shape) - 1)),
        edge_array[index_array],
        fill_value,
    )
