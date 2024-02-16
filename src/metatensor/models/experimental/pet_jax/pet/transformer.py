from typing import List, Optional

import equinox as eqx
import jax
import jax.numpy as jnp

from .attention import AttentionBlock
from .feedforward import FeedForwardBlock


class TransformerLayer(eqx.Module):
    """A single transformer layer."""

    attention_block: AttentionBlock
    ff_block: FeedForwardBlock

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        dropout_rate: float,
        attention_dropout_rate: float,
        key: jax.random.PRNGKey,
    ):
        attention_key, ff_key = jax.random.split(key)

        self.attention_block = AttentionBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            key=attention_key,
        )
        self.ff_block = FeedForwardBlock(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dropout_rate=dropout_rate,
            key=ff_key,
        )

    def __call__(
        self,
        inputs: jnp.ndarray,  # seq_len hidden_size
        radial_mask: jnp.ndarray,  # seq_len
        enable_dropout: bool = False,
        key: Optional[jax.random.PRNGKey] = None,
    ) -> jnp.ndarray:  # seq_len hidden_size

        attn_key, ff_key = (None, None) if key is None else jax.random.split(key)
        attention_output = self.attention_block(
            inputs, radial_mask, enable_dropout=enable_dropout, key=attn_key
        )
        seq_len = inputs.shape[0]
        ff_keys = None if ff_key is None else jax.random.split(ff_key, num=seq_len)
        output = jax.vmap(self.ff_block, in_axes=(0, None, 0))(
            attention_output, enable_dropout, ff_keys
        )

        return output


class Transformer(eqx.Module):
    """A transformer model."""

    layers: List[eqx.Module]

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_layers: int,
        dropout_rate: float,
        attention_dropout_rate: float,
        key: jax.random.PRNGKey,
    ):

        keys = jax.random.split(key, num=num_layers)
        self.layers = [
            TransformerLayer(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                attention_dropout_rate=attention_dropout_rate,
                key=layer_key,
            )
            for layer_key in keys
        ]

    def __call__(
        self,
        inputs: jnp.ndarray,  # seq_len hidden_size
        enable_dropout: bool,
        radial_mask: jnp.ndarray,  # seq_len
        key: Optional[jax.random.PRNGKey] = None,
    ) -> jnp.ndarray:  # seq_len hidden_size

        x = inputs

        for layer in self.layers:
            current_key, key = (None, None) if key is None else jax.random.split(key)
            x = layer(x, radial_mask, enable_dropout=enable_dropout, key=current_key)

        return x
