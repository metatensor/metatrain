import jax
import jax.numpy as jnp

from metatensor.models.experimental.pet_jax.pet.utils.corresponding_edges import (
    get_corresponding_edges,
)
from metatensor.models.experimental.pet_jax.pet.utils.edges_to_nef import (
    edge_array_to_nef,
    get_nef_indices,
    nef_array_to_edges,
)


def test_corresponding_edges():
    """Tests the get_corresponding_edges function, needed for message passing."""

    get_corresponding_edges_jit = jax.jit(get_corresponding_edges)
    arr = jnp.array([[0, 1]] * 500 + [[1, 0]] * 500)
    corresponding_edges = get_corresponding_edges_jit(arr)
    expected = jnp.array([500] * 500 + [0] * 500)
    assert jnp.all(corresponding_edges == expected)


def test_nef_indices():
    """Tests the NEF indexing, needed to feed edges to a transformer."""

    get_nef_indices_jit = jax.jit(get_nef_indices, static_argnums=(1, 2))
    edge_array_to_nef_jit = jax.jit(edge_array_to_nef)
    nef_array_to_edges_jit = jax.jit(nef_array_to_edges)

    centers = jnp.array([0, 4, 3, 1, 0, 0, 3, 3, 3, 4])
    nef_indices, nef_to_edges_neighbor, nef_mask = get_nef_indices_jit(centers, 5, 4)

    expected_nef_mask = jnp.array(
        [
            [True, True, True, False],
            [True, False, False, False],
            [False, False, False, False],
            [True, True, True, True],
            [True, True, False, False],
        ]
    )
    assert jnp.all(nef_mask == expected_nef_mask)

    nef_centers = edge_array_to_nef_jit(centers, nef_indices)

    expected_nef_centers = jnp.array(
        [[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [3, 3, 3, 3], [4, 4, 0, 0]]
    )

    assert jnp.all(nef_centers == expected_nef_centers)

    centers_again = nef_array_to_edges_jit(nef_centers, centers, nef_to_edges_neighbor)
    assert jnp.all(centers == centers_again)
