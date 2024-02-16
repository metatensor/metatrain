import equinox as eqx
import jax
import jax.numpy as jnp

from .radial_attention import RadialAttention


class AttentionBlock(eqx.Module):
    """A single transformer attention block."""

    attention: RadialAttention
    layernorm: eqx.nn.Embedding
    dropout: eqx.nn.Dropout
    num_heads: int = eqx.static_field

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout_rate: float,
        attention_dropout_rate: float,
        key: jax.random.PRNGKey,
    ):
        self.num_heads = num_heads
        self.attention = RadialAttention(
            num_heads=num_heads,
            query_size=hidden_size,
            use_query_bias=True,
            use_key_bias=True,
            use_value_bias=True,
            use_output_bias=True,
            dropout_p=attention_dropout_rate,
            key=key,
        )
        self.layernorm = eqx.nn.LayerNorm(shape=hidden_size)
        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(
        self,
        inputs: jnp.ndarray,  # seq_len hidden_size
        radial_mask: jnp.ndarray,  # seq_len
        enable_dropout: bool = False,
        key: "jax.random.PRNGKey" = None,
    ) -> jnp.ndarray:  # seq_len hidden_size

        attention_key, dropout_key = (
            (None, None) if key is None else jax.random.split(key)
        )

        # Pre-layer normalization
        normed_inputs = jax.vmap(self.layernorm)(inputs)

        # Attention
        attention_output = self.attention(
            query=inputs,
            key_=inputs,
            value=inputs,
            radial_mask=radial_mask,
            inference=not enable_dropout,
            key=attention_key,
        )

        # Apply dropout
        output = self.dropout(
            attention_output, inference=not enable_dropout, key=dropout_key
        )

        # Residual connection
        output += inputs

        return output
