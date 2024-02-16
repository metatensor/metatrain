import jax
import jax.numpy as jnp


def get_radial_mask(r, r_cut, r_transition):
    # All radii are already guaranteed to be smaller than r_cut
    return jnp.where(
        r < r_transition,
        jnp.ones_like(r),
        0.5 * (jnp.cos(jnp.pi * (r - r_transition) / (r_cut - r_transition)) + 1.0),
    )
