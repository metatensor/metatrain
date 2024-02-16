from typing import Dict

import equinox as eqx
import jax
import jax.numpy as jnp


class Encoder(eqx.Module):

    cartesian_encoder: eqx.nn.Linear
    center_encoder: eqx.nn.Embedding
    neighbor_encoder: eqx.nn.Embedding
    compressor: eqx.nn.Linear

    def __init__(
        self,
        n_species: int,
        hidden_size: int,
        key: jax.random.PRNGKey,
    ):
        key1, key2, key3 = jax.random.split(key, num=3)

        self.cartesian_encoder = eqx.nn.Linear(
            in_features=3, out_features=hidden_size, key=key1
        )
        self.center_encoder = eqx.nn.Embedding(
            num_embeddings=n_species, embedding_size=hidden_size, key=key2
        )
        self.neighbor_encoder = eqx.nn.Embedding(
            num_embeddings=n_species, embedding_size=hidden_size, key=key3
        )
        self.compressor = eqx.nn.Linear(
            in_features=3 * hidden_size, out_features=hidden_size, key=key
        )

    def __call__(
        self,
        features: Dict[str, jnp.ndarray],
    ):
        # Encode cartesian coordinates
        cartesian_features = jax.vmap(jax.vmap(self.cartesian_encoder))(
            features["cartesian"]
        )

        # Encode centers
        center_features = jax.vmap(jax.vmap(self.center_encoder))(features["center"])

        # Encode neighbors
        neighbor_features = jax.vmap(jax.vmap((self.neighbor_encoder)))(
            features["neighbor"]
        )

        # Concatenate
        encoded_features = jnp.concatenate(
            [cartesian_features, center_features, neighbor_features], axis=-1
        )

        # Compress
        compressed_features = jax.vmap(jax.vmap(self.compressor))(encoded_features)

        return compressed_features
