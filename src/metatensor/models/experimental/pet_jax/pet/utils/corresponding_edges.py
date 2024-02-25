import functools as ft

import jax
import jax.numpy as jnp


def loop_body(i, carry):
    array, array_inversed, inverse_indices = carry
    inverse_indices = inverse_indices.at[i].set(
        jnp.nonzero(array_inversed == array[i], size=1)[0][0]
    )
    return array, array_inversed, inverse_indices


def get_corresponding_edges(array):
    n_edges = len(array)
    int_dtype = jnp.int64 if jax.config.jax_enable_x64 else jnp.int32
    array_inversed = array[:, ::-1]
    return jax.lax.fori_loop(
        0,
        n_edges,
        loop_body,
        (array, array_inversed, jnp.empty((n_edges,), dtype=int_dtype)),
    )[2]
