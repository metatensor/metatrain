from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp


class FeedForwardBlock(eqx.Module):
    """A single transformer feed forward block."""

    mlp: eqx.nn.Linear
    output: eqx.nn.Linear
    layernorm: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout_rate: float,
        key: jax.random.PRNGKey,
    ):
        mlp_key, output_key = jax.random.split(key)
        self.mlp = eqx.nn.Linear(
            in_features=hidden_size, out_features=intermediate_size, key=mlp_key
        )
        self.output = eqx.nn.Linear(
            in_features=intermediate_size, out_features=hidden_size, key=output_key
        )

        self.layernorm = eqx.nn.LayerNorm(shape=hidden_size)
        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(
        self,
        inputs: jnp.ndarray,  # hidden_size
        enable_dropout: bool = True,
        key: Optional[jax.random.PRNGKey] = None,
    ) -> jnp.ndarray:  # hidden_size

        # Pre-layer normalization
        normed_inputs = self.layernorm(inputs)

        # Feed-forward
        hidden = self.mlp(normed_inputs)
        hidden = jax.nn.gelu(hidden)

        # Project back to input size
        output = self.output(hidden)

        # Apply dropout
        output = self.dropout(output, inference=not enable_dropout, key=key)

        # Residual connection
        output += inputs

        return output
