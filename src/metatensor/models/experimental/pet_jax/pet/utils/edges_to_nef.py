import functools as ft

import jax
import jax.numpy as jnp


def loop_body(i, carry):
    centers, edges_to_nef, nef_to_edges_neighbor, nef_mask, node_counter = carry
    center = centers[i]
    edges_to_nef = edges_to_nef.at[center, node_counter[center]].set(i)
    nef_mask = nef_mask.at[center, node_counter[center]].set(True)
    nef_to_edges_neighbor = nef_to_edges_neighbor.at[i].set(node_counter[center])
    node_counter = node_counter.at[center].add(1)
    return centers, edges_to_nef, nef_to_edges_neighbor, nef_mask, node_counter


def get_nef_indices(centers, n_nodes: int, n_edges_per_node: int):
    int_dtype = jnp.int64 if jax.config.jax_enable_x64 else jnp.int32
    n_edges = len(centers)
    edges_to_nef = jnp.zeros((n_nodes, n_edges_per_node), dtype=int_dtype)
    nef_to_edges_neighbor = jnp.empty((n_edges,), dtype=int_dtype)
    node_counter = jnp.zeros((n_nodes,), dtype=int_dtype)
    nef_mask = jnp.full((n_nodes, n_edges_per_node), False, dtype=bool)
    # returns edges_to_nef, nef_to_edges_neighbor, nef_mask
    # edges_to_nef can be used to index an edge array to get the corresponding nef array
    # nef_to_edges_neighbor can be used to index the second dimension of a nef array
    # to get the corresponding edge array (the first dimension is indexed by `centers`)
    # nef_mask masks out the padding values in the nef array
    return jax.lax.fori_loop(
        0,
        n_edges,
        loop_body,
        (centers, edges_to_nef, nef_to_edges_neighbor, nef_mask, node_counter),
    )[1:4]


def edge_array_to_nef(edge_array, nef_indices, mask=None, fill_value=0.0):
    if mask is None:
        return edge_array[nef_indices]
    else:
        return jnp.where(mask, edge_array[nef_indices], fill_value)


def nef_array_to_edges(nef_array, centers, nef_to_edges_neighbor):
    return nef_array[centers, nef_to_edges_neighbor]
